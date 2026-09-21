export NCCL_TIMEOUT=36000
export WANDB_API_KEY=589d531f376ac4cfc8d713753b96052a1420b709
export http_proxy=http://oversea-squid2.ko.txyun:11080
export https_proxy=http://oversea-squid2.ko.txyun:11080
export TORCH_HOME=/video_ssd/sunming03/xutongda/torchhome
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/share/xutongda/hfhome


EXPS=(
    /video_ssd/kongzishang/xutongda/git/IFID/exps_unet/unet-raet2is-400k
)

for EXP in "${EXPS[@]}"; do
    for CFG in 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0 5.25 5.5 5.75 6.0; do
    # for CFG in 1.0; do
        echo "Running EXP=${EXP}, CFG=${CFG}, STEPS=400000"

        torchrun --nnodes=1 --nproc_per_node=8 --master_port 29506 generate.py \
            --num-fid-samples 50000 \
            --pproc-batch-size 32 \
            --path-type linear \
            --mode sde \
            --num-steps 250 \
            --cfg-scale ${CFG} \
            --guidance-high 1.0 \
            --guidance-low 0.0 \
            --exp-path "$EXP" \
            --train-steps 400000 \
            --fid-reference-file "/video_ssd/kongzishang/xutongda/git/REPA-E/exps/VIRTUAL_imagenet256_labeled.npz"
    done
done