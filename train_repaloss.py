import argparse
import copy
import logging
import os
import json
import math
from pathlib import Path
from collections import OrderedDict
from omegaconf import OmegaConf
from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta, datetime
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import wandb
from ifid.fid.psnr import get_psnr
from ifid.dataset import CustomINH5Dataset
from ifid.sit.sit import SiT_models
from ifid.sit.samplers import euler_sampler
from ifid.vae.utils import instantiate_from_config, load_encoders
from PIL import Image 
from accelerate.utils import DistributedDataParallelKwargs
import lpips
import torchvision.transforms.v2 as v2
from ifid.sit.sit import mean_flat
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
import torch.nn.functional as F
import torch.nn as nn
logger = get_logger(__name__)

from torch.distributed.nn.functional import all_gather

def differentiable_all_gather(x):
    if not torch.distributed.is_initialized():
        return x
    return torch.cat(all_gather(x), dim=0)

class EMAFeatureStats(nn.Module):
    """
    EMA first/second moments for FD.

    Follows the EMA formulation of the official FD-Loss implementation:

        mu  = beta * mu_ema
              + (1-beta) * mean(current_feats)

        m2  = beta * m2_ema
              + (1-beta) * E[x x^T]

        cov = m2 - mu mu^T

    `build_stats()` is differentiable w.r.t. current features.
    `update()` modifies the EMA using detached features.
    """

    def __init__(self, feat_dim, beta=0.999):
        super().__init__()

        self.feat_dim = feat_dim
        self.beta = beta

        self.register_buffer(
            "mu_ema",
            torch.zeros(feat_dim, dtype=torch.float64)
        )

        self.register_buffer(
            "m2_ema",
            torch.zeros(
                feat_dim,
                feat_dim,
                dtype=torch.float64,
            )
        )

        self.register_buffer(
            "initialized",
            torch.tensor(False)
        )

    def build_stats(self, feats):
        """
        Compute EMA-blended statistics while preserving gradient
        through the current features.

        feats: [B, C]
        """

        feats = feats.double()

        batch_mu = feats.mean(dim=0)
        batch_m2 = feats.T @ feats / feats.shape[0]

        # First batch: just use batch stats.
        if not self.initialized.item():
            mu = batch_mu
            m2 = batch_m2

        else:
            beta = self.beta

            mu = (
                beta * self.mu_ema.detach()
                + (1.0 - beta) * batch_mu
            )

            m2 = (
                beta * self.m2_ema.detach()
                + (1.0 - beta) * batch_m2
            )

        cov = m2 - mu[:, None] * mu[None, :]

        # numerical symmetrization
        cov = 0.5 * (cov + cov.T)

        return mu, cov

    @torch.no_grad()
    def update(self, feats):
        """
        Update EMA state. Call AFTER constructing/backpropagating
        the current FD loss.

        feats should normally already be globally gathered.
        """

        feats = feats.detach().double()

        batch_mu = feats.mean(dim=0)
        batch_m2 = feats.T @ feats / feats.shape[0]

        if not self.initialized.item():
            self.mu_ema.copy_(batch_mu)
            self.m2_ema.copy_(batch_m2)
            self.initialized.fill_(True)
            return

        beta = self.beta

        self.mu_ema.mul_(beta).add_(
            batch_mu,
            alpha=1.0 - beta,
        )

        self.m2_ema.mul_(beta).add_(
            batch_m2,
            alpha=1.0 - beta,
        )

class EMAFactorizedFeatureStats(nn.Module):
    """
    EMA statistics for features [B, L, C] under

        Cov(vec(F)) ~= var * R_L kron R_C

    where
        R_L: [L, L], spatial covariance factor
        R_C: [C, C], channel covariance factor

    and approximately

        trace(R_L) = L
        trace(R_C) = C.

    We keep EMA of RAW moments:
        E[x]                  : [L, C]
        E_c[x x^T]            : [L, L]
        E_l[x^T x]            : [C, C]

    so that covariance can be reconstructed correctly.
    """

    def __init__(self, beta=0.999, eps=1e-8):
        super().__init__()

        self.beta = beta
        self.eps = eps

        # Lazy initialization because DINO and teacher
        # can have different L / C.
        self.register_buffer("mu_ema", None)
        self.register_buffer("m2_L_ema", None)
        self.register_buffer("m2_C_ema", None)

        self.register_buffer(
            "initialized",
            torch.tensor(False),
        )

    def _batch_raw_moments(self, feats):
        """
        feats: [B, L, C]

        Returns:
            mu       [L, C]
            m2_L     [L, L]
            m2_C     [C, C]
        """
        feats = feats.double()

        B, L, C = feats.shape

        # E_b[x]
        mu = feats.mean(dim=0)                     # [L, C]

        # E_{b,c}[ x_lc x_mc ]
        m2_L = torch.einsum(
            "blc,bmc->lm",
            feats,
            feats,
        ) / (B * C)

        # E_{b,l}[ x_blc x_bld ]
        m2_C = torch.einsum(
            "blc,bld->cd",
            feats,
            feats,
        ) / (B * L)

        return mu, m2_L, m2_C

    def _moments_to_stats(
        self,
        mu,
        m2_L,
        m2_C,
    ):
        """
        Convert raw moments to

            mean  : [L, C]
            var   : scalar
            R_L   : [L, L]
            R_C   : [C, C]
        """

        L, C = mu.shape

        # --------------------------------------------------
        # Spatial covariance
        #
        # E[x_l x_m] - E[x_l] E[x_m]
        # averaged over channels.
        # --------------------------------------------------
        mu_outer_L = torch.einsum(
            "lc,mc->lm",
            mu,
            mu,
        ) / C

        cov_L = m2_L - mu_outer_L
        cov_L = 0.5 * (cov_L + cov_L.T)

        # --------------------------------------------------
        # Channel covariance
        #
        # E[x_c x_d] - E[x_c] E[x_d]
        # averaged over positions.
        # --------------------------------------------------
        mu_outer_C = torch.einsum(
            "lc,ld->cd",
            mu,
            mu,
        ) / L

        cov_C = m2_C - mu_outer_C
        cov_C = 0.5 * (cov_C + cov_C.T)

        # --------------------------------------------------
        # Overall variance.
        #
        # trace(m2_L) / L = E[x^2]
        # --------------------------------------------------
        second_moment = torch.trace(m2_L) / L
        mean_square = mu.square().mean()

        var = (second_moment - mean_square).clamp_min(
            self.eps
        )

        # --------------------------------------------------
        # Normalize factors so approximately:
        #
        # trace(R_L) = L
        # trace(R_C) = C
        #
        # Both cov_L and cov_C have average diagonal = var.
        # --------------------------------------------------
        R_L = cov_L / var
        R_C = cov_C / var

        # numerical symmetrization
        R_L = 0.5 * (R_L + R_L.T)
        R_C = 0.5 * (R_C + R_C.T)

        return mu, var, R_L, R_C

    def build_stats(self, feats):
        """
        Differentiable current statistics:

            EMA = beta * old_ema
                  + (1-beta) * current_batch

        Current batch remains differentiable.

        feats: [B, L, C]
        """

        assert feats.ndim == 3

        (
            batch_mu,
            batch_m2_L,
            batch_m2_C,
        ) = self._batch_raw_moments(feats)

        if not self.initialized.item():

            mu = batch_mu
            m2_L = batch_m2_L
            m2_C = batch_m2_C

        else:

            beta = self.beta

            mu = (
                beta * self.mu_ema.detach()
                + (1.0 - beta) * batch_mu
            )

            m2_L = (
                beta * self.m2_L_ema.detach()
                + (1.0 - beta) * batch_m2_L
            )

            m2_C = (
                beta * self.m2_C_ema.detach()
                + (1.0 - beta) * batch_m2_C
            )

        return self._moments_to_stats(
            mu,
            m2_L,
            m2_C,
        )

    @torch.no_grad()
    def update(self, feats):
        """
        Update persistent EMA state.

        feats should already contain GLOBAL features
        gathered across GPUs.
        """

        feats = feats.detach().double()

        (
            batch_mu,
            batch_m2_L,
            batch_m2_C,
        ) = self._batch_raw_moments(feats)

        if not self.initialized.item():

            # lazy initialization
            self.mu_ema = batch_mu.clone()
            self.m2_L_ema = batch_m2_L.clone()
            self.m2_C_ema = batch_m2_C.clone()

            self.initialized.fill_(True)
            return

        beta = self.beta

        self.mu_ema.mul_(beta).add_(
            batch_mu,
            alpha=1.0 - beta,
        )

        self.m2_L_ema.mul_(beta).add_(
            batch_m2_L,
            alpha=1.0 - beta,
        )

        self.m2_C_ema.mul_(beta).add_(
            batch_m2_C,
            alpha=1.0 - beta,
        )

def bures_affinity(A, B, eps=1e-8):
    """
    Tr[
        (A^{1/2} B A^{1/2})^{1/2}
    ]

    A, B: [D, D] PSD matrices
    """

    D = A.shape[0]

    eye = torch.eye(
        D,
        device=A.device,
        dtype=A.dtype,
    )

    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)

    A = A + eps * eye
    B = B + eps * eye

    # A^{1/2}
    evals_A, evecs_A = torch.linalg.eigh(A)

    evals_A = evals_A.clamp_min(0)

    A_sqrt = (
        evecs_A
        * evals_A.sqrt().unsqueeze(0)
    ) @ evecs_A.T

    # A^{1/2} B A^{1/2}
    middle = (
        A_sqrt
        @ B
        @ A_sqrt
    )

    middle = 0.5 * (
        middle + middle.T
    )

    evals_middle = torch.linalg.eigvalsh(
        middle
    ).clamp_min(0)

    return evals_middle.sqrt().sum()

def factorized_frechet_from_stats(
    mu_pred,
    var_pred,
    R_L_pred,
    R_C_pred,
    mu_ref,
    var_ref,
    R_L_ref,
    R_C_ref,
    eps=1e-8,
):
    """
    FD under

        Sigma ~= var * R_L kron R_C

    mu_*:   [L, C]
    R_L_*:  [L, L]
    R_C_*:  [C, C]
    """

    L, C = mu_pred.shape

    # ======================================================
    # Mean term
    #
    # This is exactly the flattened [L*C] mean distance.
    # ======================================================
    mean_term = (
        mu_pred - mu_ref
    ).square().sum()

    # ======================================================
    # Bures affinities of two factors
    # ======================================================
    affinity_L = bures_affinity(
        R_L_ref,
        R_L_pred,
        eps=eps,
    )

    affinity_C = bures_affinity(
        R_C_ref,
        R_C_pred,
        eps=eps,
    )

    # Since
    #
    #   trace(R_L) ~= L
    #   trace(R_C) ~= C
    #
    # trace(Sigma) ~= var * L * C
    #
    trace_pred = var_pred * L * C
    trace_ref = var_ref * L * C

    # Kronecker property:
    #
    # affinity(Sigma_r, Sigma_p)
    #
    # = sqrt(var_r * var_p)
    #   * affinity(RL_r, RL_p)
    #   * affinity(RC_r, RC_p)
    #
    cross = (
        torch.sqrt(
            (var_pred * var_ref).clamp_min(0)
            + eps
        )
        * affinity_L
        * affinity_C
    )

    covariance_term = (
        trace_pred
        + trace_ref
        - 2.0 * cross
    )

    return mean_term + covariance_term


def ema_factorized_fd_loss(
    feat_pred,
    feat_ref,
    pred_stats,
    ref_stats,
):
    """
    feat_pred: [B, L, C]
    feat_ref:  [B, L, C]

    Compute global multi-GPU EMA FD assuming

        Cov(vec(F))
            ~= var * R_L kron R_C

    No spatial/token pooling is performed.
    """

    assert feat_pred.ndim == 3
    assert feat_ref.ndim == 3

    assert feat_pred.shape[1:] == feat_ref.shape[1:]

    # ======================================================
    # Gather across GPUs.
    #
    # [B_local, L, C]
    #       ->
    # [B_global, L, C]
    # ======================================================

    feat_pred_global = differentiable_all_gather(
        feat_pred.float()
    )

    feat_ref_global = differentiable_all_gather(
        feat_ref.float()
    )

    # ======================================================
    # Differentiable EMA stats
    # ======================================================

    (
        mu_pred,
        var_pred,
        R_L_pred,
        R_C_pred,
    ) = pred_stats.build_stats(
        feat_pred_global
    )

    (
        mu_ref,
        var_ref,
        R_L_ref,
        R_C_ref,
    ) = ref_stats.build_stats(
        feat_ref_global
    )

    # ======================================================
    # Factorized FD
    # ======================================================

    loss = factorized_frechet_from_stats(
        mu_pred,
        var_pred,
        R_L_pred,
        R_C_pred,

        mu_ref,
        var_ref,
        R_L_ref,
        R_C_ref,
    )

    return (
        loss,
        feat_pred_global,
        feat_ref_global,
    )

def frechet_from_stats(
    mu_pred,
    cov_pred,
    mu_ref,
    cov_ref,
    eps=1e-8,
):
    """
    Differentiable FD from mean/covariance.

    All inputs preferably float64.
    """

    mean_term = (mu_pred - mu_ref).square().sum()

    C = cov_pred.shape[0]

    eye = torch.eye(
        C,
        device=cov_pred.device,
        dtype=cov_pred.dtype,
    )

    cov_pred = cov_pred + eps * eye
    cov_ref = cov_ref + eps * eye

    # sqrt(cov_ref)
    evals_ref, evecs_ref = torch.linalg.eigh(cov_ref)

    evals_ref = evals_ref.clamp_min(0)

    cov_ref_sqrt = (
        evecs_ref
        * evals_ref.sqrt().unsqueeze(0)
    ) @ evecs_ref.T

    middle = (
        cov_ref_sqrt
        @ cov_pred
        @ cov_ref_sqrt
    )

    middle = 0.5 * (middle + middle.T)

    evals_middle = torch.linalg.eigvalsh(
        middle
    ).clamp_min(0)

    covariance_term = (
        torch.trace(cov_pred)
        + torch.trace(cov_ref)
        - 2.0 * evals_middle.sqrt().sum()
    )

    return mean_term + covariance_term


def ema_fd_loss(
    feat_pred,
    feat_ref,
    pred_stats,
    ref_stats,
):
    """
    feat_pred / feat_ref:
        [B, C] or [B, L, C]

    Uses GLOBAL batch across GPUs.
    """

    # [B, L, C] -> [B, C]
    if feat_pred.ndim == 3:
        feat_pred = feat_pred.mean(dim=1)

    if feat_ref.ndim == 3:
        feat_ref = feat_ref.mean(dim=1)

    # Global generated features, differentiable
    feat_pred_global = differentiable_all_gather(
        feat_pred.float()
    )

    # Global reference features, no gradient needed
    feat_ref_global = differentiable_all_gather(
        feat_ref.float()
    )

    # Current loss uses:
    #
    # beta * previous EMA
    # +
    # (1-beta) * CURRENT batch
    #
    # Thus gradient flows through feat_pred_global.
    mu_pred, cov_pred = pred_stats.build_stats(
        feat_pred_global
    )

    mu_ref, cov_ref = ref_stats.build_stats(
        feat_ref_global
    )

    loss = frechet_from_stats(
        mu_pred,
        cov_pred,
        mu_ref,
        cov_ref,
    )

    return (
        loss,
        feat_pred_global,
        feat_ref_global,
    )

def count_trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def preprocess_imgs_vae(imgs):
    # imgs: (B, C, H, W) -> (B, C, H, W), [0, 255] uint8 -> [-1, 1] float32
    return imgs.float() / 127.5 - 1.0


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = (
        x.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    return x


def sample_posterior(moments, latents_scale=1.0, latents_bias=0.0):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z - latents_bias) * latents_scale  # normalize
    return z


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    # Also perform EMA on BN buffers
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if buffer.dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float64,
        ):
            # Apply EMA only to float buffers
            ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
        else:
            # Direct copy for non-float buffers
            ema_buffers[name].copy_(buffer)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    else:
        raise NotImplementedError

    return x


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    assert args.gradient_accumulation_steps == 1
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, timeout_kwargs],
    )

    # set up the logger and checkpoint dirs
    if accelerator.is_main_process:
        os.makedirs(
            args.output_dir, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, "w") as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    torch.backends.cudnn.benchmark = True

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    # Create model:
    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for name, param in vae.named_parameters():
        param.requires_grad_(False)

    if args.enc_type != None:
        with accelerator.local_main_process_first():
            encoders, encoder_types, architectures = load_encoders(
                args.enc_type, device, args.resolution
            )
    else:
        raise NotImplementedError()

    z_dims = [1024]

    fd_beta = 0.999

    if args.proj_type == 'fd_dino' or args.proj_type == 'fd_repa':
        fd_dino_pred_stats = EMAFeatureStats(
            feat_dim=1024,
            beta=fd_beta,
        ).to(device)
        fd_dino_ref_stats = EMAFeatureStats(
            feat_dim=1024,
            beta=fd_beta,
        ).to(device)
    elif args.proj_type == 'fd_dino_fact':
        fd_dino_pred_stats = EMAFactorizedFeatureStats(
            beta=fd_beta,
        ).to(device)
        fd_dino_ref_stats = EMAFactorizedFeatureStats(
            beta=fd_beta,
        ).to(device)
    else:
        pass

    fake_in = torch.zeros([1, 3, args.resolution, args.resolution]).to(device)
    fake_z = vae.encode(fake_in)[0]

    if accelerator.is_main_process:
        logger.info(f"VAE fake_z shape: {tuple(fake_z.shape)}")
        logger.info(f"VAE fake_z numel: {fake_z.numel()}")
        logger.info(f"VAE fake_z mean/std: {fake_z.mean().item():.4f}/{fake_z.std().item():.4f}")

    if len(fake_z.shape) == 3:
        # 2d latent
        latent_size = fake_z.shape[-1]
        in_channels = fake_z.shape[0]
        xT = torch.randn((n, in_channels, latent_size, latent_size), device=device)
    elif len(fake_z.shape) == 2:
        # 1d latent
        latent_size = fake_z.shape[0]
        in_channels = fake_z.shape[-1]
        xT = torch.randn((n, latent_size, in_channels), device=device)
    else:
        assert 0

    tshift = math.sqrt(float(fake_z.numel()) / 4096.0)

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        class_dropout_prob=args.cfg_prob,
        bn_momentum=args.bn_momentum,
        tshift=tshift,
        prediction_internal=args.prediction_internal,
        tk_drop=args.token_drop,
        tk_drop_mode=args.token_drop_mode,
        tk_drop_param=args.token_drop_param,
        tk_drop_dim=args.token_drop_dim,
        z_dims=z_dims,
        **block_kwargs,
    )

    # make a copy of the model for EMA
    model = model.to(device)
    ema = copy.deepcopy(model).to(
        device
    )  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    teacher = copy.deepcopy(model).to(device)
    # teacher.encoder_depth = 4
    requires_grad(teacher, False)

    # Apply SyncBN if more than 1 GPU is used
    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Define the optimizers for SiT, VAE, and VAE loss function separately
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = CustomINH5Dataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    # Start with eval mode for all models
    model.eval()
    ema.eval()
    vae.eval()

    # resume
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) + ".pt"
        ckpt_path = f"{args.cont_dir}/checkpoints/{ckpt_name}"

        # If the checkpoint exists, we load the checkpoint and resume the training
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        (optimizer.load_state_dict(ckpt["opt"]),)
        global_step = ckpt["steps"]

    teacher_ckpt_name = str(400000).zfill(7) + ".pt"
    teacher_ckpt_path = f"{args.teacher_dir}/checkpoints/{teacher_ckpt_name}"
    teacher_ckpt = torch.load(teacher_ckpt_path, map_location="cpu", weights_only=False)
    teacher.load_state_dict(teacher_ckpt["ema"])

    # Allow larger cache size for DYNAMo compilation
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    # Model compilation for better performance

    model, vae, optimizer, train_dataloader = accelerator.prepare(
        model, vae, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="diffusion vae arena",
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"{args.exp_name}"}},
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(args.epochs):
        model.train()

        for raw_image, y in train_dataloader:
            raw_image = raw_image.to(device)
            labels = y.to(device)

            with torch.no_grad():
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)

            model.train()
            requires_grad(model, True)

            with accelerator.autocast():
                # 1). Forward pass: VAE
                processed_image = preprocess_imgs_vae(raw_image)

                vae.eval()
                with torch.no_grad():
                    z = vae.encode(processed_image)
                    xhat = vae.decode(z)

                # 2). Backward pass: VAE, compute the VAE loss, backpropagate, and update the VAE; Then, compute the riminator loss and update the discriminator
                #    loss_kwargs used for SiT forward function, create here and can be reused for both VAE and SiT
                loss_kwargs = dict(
                    path_type=args.path_type,
                    prediction=args.prediction,
                    weighting=args.weighting,
                )
                # Record the time_input and noises for the VAE alignment, so that we avoid sampling again
                time_in = torch.rand((z.shape[0], 1, 1, 1))
                # noises = torch.randn_like(z)
                noises = None

                # 3). Forward pass: SiT
                # **Avoid diffusion loss to backpropagate to the VAE, so we detach the `z`**
                loss_kwargs["align_only"] = False

                sit_outputs = model(
                    x=z,
                    y=labels,
                    zs=zs,
                    loss_kwargs=loss_kwargs,
                    time_input=time_in,
                    noises=None,
                    return_feat=False,
                )

                sit_loss = sit_outputs["denoising_loss"].mean()

                loss_kwargs["align_only"] = True

                if args.proj_type == "cos":
                    # tau = 0.8
                    # z_t = z * tau + sit_outputs["pred_x"] * (1 - tau)
                    time_t = time_in * 0.25
                    # a_t = ((time_t * (1 - time_in)) / (time_in * (1 - time_t) + 1e-3)).to(z.device)
                    # correction
                    z_t = sit_outputs["pred_x"]
                    # z_t = vae.encode(vae.decode(z_t)) # idempotence
                    # z_t = a_t * z + (1 - a_t) * sit_outputs["pred_x"]
                    teacher_outputs = ema(
                        x=torch.cat([z, z_t], dim=0),
                        y=torch.cat([labels, labels], dim=0),
                        loss_kwargs=loss_kwargs,
                        time_input=torch.cat([time_t, time_t], dim=0),
                        noises=None,
                        return_feat=True,
                    )
                    feat_ref, feat_dis = torch.chunk(teacher_outputs["fs_tilde"][0], 2, dim=0)
                    feat_ref, feat_dis = F.normalize(feat_ref, dim=-1), F.normalize(feat_dis, dim=-1)
                    proj_loss = torch.mean(mean_flat(-(feat_ref * feat_dis).sum(dim=-1)))
                elif args.proj_type == "mse":
                    teacher_outputs = teacher(
                        x=torch.cat([z, sit_outputs["pred_x"]], dim=0),
                        y=torch.cat([labels, labels], dim=0),
                        loss_kwargs=loss_kwargs,
                        time_input=torch.cat([time_in, time_in], dim=0),
                        noises=None,
                        return_feat=True,
                    )
                    feat_ref, feat_dis = torch.chunk(teacher_outputs["fs_tilde"][0], 2, dim=0)
                    proj_loss = torch.mean((feat_ref - feat_dis)**2)
                elif args.proj_type == "repa":
                    tau = 0.8
                    z_t = z * tau + sit_outputs["pred_x"] * (1 - tau)
                    time_t = time_in * tau
                    # time_t = (time_in + 1.0) * 0.5
                    teacher_outputs = teacher(
                        x=z_t,
                        y=labels,
                        zs=zs,
                        loss_kwargs=loss_kwargs,
                        time_input=time_t,
                        noises=None,
                        return_feat=False,
                    )
                    proj_loss = teacher_outputs["proj_loss"]
                elif args.proj_type == 'dino':
                    x_t = ((vae.decode(sit_outputs["pred_x"]) + 1.0) / 2.0) * 255.0
                    z_dis = encoder.forward_features(preprocess_raw_image(x_t, encoder_type))['x_norm_patchtokens']
                    z_ref = zs[0]
                    z_dis = F.normalize(z_dis, dim=-1) 
                    z_ref = F.normalize(z_ref, dim=-1) 
                    proj_loss = mean_flat(-(z_dis * z_ref).sum(dim=-1))
                    proj_loss = torch.mean(proj_loss)
                elif args.proj_type == 'fd_dino':
                    x_t = ((vae.decode(sit_outputs["pred_x"]) + 1.0) / 2.0) * 255.0
                    z_dis = encoder.forward_features(preprocess_raw_image(x_t, encoder_type))['x_norm_patchtokens']
                    z_ref = zs[0]
                    (
                        proj_loss,
                        fd_pred_feats,
                        fd_ref_feats,
                    ) = ema_fd_loss(
                        z_dis,
                        z_ref,
                        fd_dino_pred_stats,
                        fd_dino_ref_stats,
                    )
                    fd_dino_pred_stats.update(
                        fd_pred_feats
                    )
                    fd_dino_ref_stats.update(
                        fd_ref_feats
                    )
                elif args.proj_type == 'fd_repa':
                    time_t = time_in * 0.25
                    z_t = sit_outputs["pred_x"]
                    teacher_outputs = ema(
                        x=torch.cat([z, z_t], dim=0),
                        y=torch.cat([labels, labels], dim=0),
                        loss_kwargs=loss_kwargs,
                        time_input=torch.cat([time_t, time_t], dim=0),
                        noises=None,
                        return_feat=True,
                    )
                    feat_ref, feat_dis = torch.chunk(teacher_outputs["fs_tilde"][0], 2, dim=0)
                    (
                        proj_loss,
                        fd_pred_feats,
                        fd_ref_feats,
                    ) = ema_fd_loss(
                        feat_dis,
                        feat_ref,
                        fd_dino_pred_stats,
                        fd_dino_ref_stats,
                    )
                    fd_dino_pred_stats.update(
                        fd_pred_feats
                    )
                    fd_dino_ref_stats.update(
                        fd_ref_feats
                    )
                elif args.proj_type == 'fd_dino_fact':
                    x_t = ((vae.decode(sit_outputs["pred_x"]) + 1.0) / 2.0) * 255.0
                    z_dis = encoder.forward_features(preprocess_raw_image(x_t, encoder_type))['x_norm_patchtokens']
                    z_ref = zs[0]
                    (
                        proj_loss,
                        fd_pred_feats,
                        fd_ref_feats,
                    ) = ema_factorized_fd_loss(
                        z_dis,
                        z_ref,
                        fd_dino_pred_stats,
                        fd_dino_ref_stats,
                    )
                    fd_dino_pred_stats.update(
                        fd_pred_feats
                    )
                    fd_dino_ref_stats.update(
                        fd_ref_feats
                    )
                    proj_loss = proj_loss / 256.0
                elif args.proj_type == 'fd_repa_fact':
                    time_t = time_in * 0.25
                    z_t = sit_outputs["pred_x"]
                    teacher_outputs = ema(
                        x=torch.cat([z, z_t], dim=0),
                        y=torch.cat([labels, labels], dim=0),
                        loss_kwargs=loss_kwargs,
                        time_input=torch.cat([time_t, time_t], dim=0),
                        noises=None,
                        return_feat=True,
                    )
                    feat_ref, feat_dis = torch.chunk(teacher_outputs["fs_tilde"][0], 2, dim=0)
                    (
                        proj_loss,
                        fd_pred_feats,
                        fd_ref_feats,
                    ) = ema_factorized_fd_loss(
                        feat_dis,
                        feat_ref,
                        fd_dino_pred_stats,
                        fd_dino_ref_stats,
                    )
                    fd_dino_pred_stats.update(
                        fd_pred_feats
                    )
                    fd_dino_ref_stats.update(
                        fd_ref_feats
                    )
                    proj_loss = proj_loss / 256.0
                else:
                    assert(0)

                # diffusion_loss = sit_loss + proj_loss * args.proj_coeff + sit_outputs["proj_loss"] * args.proj_coeff
                diffusion_loss = sit_loss + proj_loss * args.proj_coeff
                accelerator.backward(diffusion_loss)

                if accelerator.sync_gradients:
                    grad_norm_sit = accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                psnr = torch.mean(get_psnr(processed_image, xhat, zero_mean=True, integer=True))

                # 5). Update SiT EMA
                if accelerator.sync_gradients:
                    unwrapped_model = accelerator.unwrap_model(model)
                    update_ema(
                        ema,
                        unwrapped_model._orig_mod if args.compile else unwrapped_model,
                    )

            # enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Prepare the logs based on the current step
                logs = {
                    "sit_loss": accelerator.gather(sit_loss).mean().detach().item(),
                    "proj_loss": accelerator.gather(proj_loss).mean().detach().item(),
                    "denoising_loss": accelerator.gather(sit_outputs["denoising_loss"]).mean().detach().item(),
                    "grad_norm_sit": accelerator.gather(grad_norm_sit).mean().detach().item(),
                    "epoch": epoch,
                    "psnr": accelerator.gather(psnr).mean().detach().item(),
                }

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # `model` and `vae` are wrapped by the `accelerator` object, so we need to unwrap them
                    unwrapped_model = accelerator.unwrap_model(model)

                    # model might be compiled, we extract the original model
                    original_model = (
                        unwrapped_model._orig_mod if args.compile else unwrapped_model
                    )
                    checkpoint = {
                        "model": original_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if global_step == 1 or (
                global_step % args.sampling_steps == 0 and global_step > 0
            ):
                # NOTE: Inference should use eval mode
                model.eval()
                vae.eval()
                with torch.no_grad():
                    unwrapped_model = accelerator.unwrap_model(model)
                    samples = euler_sampler(
                        unwrapped_model,
                        xT,
                        ys,
                        num_steps=50,
                        cfg_scale=2.0,
                        guidance_low=0.0,
                        guidance_high=1.0,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    samples = unwrapped_model.denormalize_latents(samples)
                    samples = accelerator.unwrap_model(vae).decode(
                        samples
                    )
                    samples = (samples + 1) / 2.0
                out_samples = accelerator.gather(samples.to(torch.float32))

                accelerator.log({"samples": wandb.Image(array2grid(out_samples))})

                logging.info("Generating EMA samples done.")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging params
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--continue-train-exp-dir", type=str, default=None)
    parser.add_argument("--wandb-history-path", type=str, default=None)

    # SiT model params
    parser.add_argument(
        "--model",
        type=str,
        default="SiT-XL/2",
        choices=SiT_models.keys(),
        help="The model to train.",
    )
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument(
        "--qk-norm", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fused-attn", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--bn-momentum", type=float, default=0.1)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to compile the model for faster training",
    )

    # dataset params
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--resolution", type=int, choices=[256], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision params
    parser.add_argument(
        "--allow-tf32", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument(
        "--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"]
    )

    # optimization params
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam-weight-decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam-epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max-grad-norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max-grad-norm-vae", default=1.0, type=float, help="Max gradient norm."
    )

    # seed params
    parser.add_argument("--seed", type=int, default=0)

    # cpu params
    parser.add_argument("--num-workers", type=int, default=4)

    # loss params
    parser.add_argument(
        "--path-type", type=str, default="linear", choices=["linear", "cosine"]
    )
    parser.add_argument(
        "--prediction",
        type=str,
        default="v",
        choices=["v"],
        help="currently we only support v-prediction",
    )
    parser.add_argument(
        "--prediction-internal",
        type=str,
        default="v",
        choices=["v", "x"],
        help="internal prediction type, could be x or v, only used for loss computation and does not affect the model architecture",
    )
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument(
        "--weighting",
        default="uniform",
        type=str,
        choices=["uniform", "lognormal"],
        help="Loss weihgting, uniform or lognormal",
    )

    # vae params
    parser.add_argument("--vae-config", type=str, default="")
    parser.add_argument("--token-drop", type=int, default=-1)
    parser.add_argument("--token-drop-mode", type=str, default="drop")
    parser.add_argument("--token-drop-param", type=float, default=1.0)
    parser.add_argument("--token-drop-dim", type=int, default=1)
    parser.add_argument("--proj-type", type=str, default="cos")
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--teacher-dir", type=str, default="")
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-l')

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)