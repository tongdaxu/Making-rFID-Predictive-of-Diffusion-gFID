# Making Reconstruction FID Predictive of Diffusion Generation FID
## Brief
* Reconstruction FID of VAE are often negatively correlated with generation FID of latent diffusion.
* We slightly change the rFID computation into interpolated FID (iFID) to make it highly correlated to gFID.

## Installation
* install by pip
    ```bash
    git clone ...
    cd ...
    pip install -e .
    ```

## VAE Arena for Diffusion Generation
* We train SiT-B, and SiT-XL model on ImageNet for 40 epoch, and evaluate the gFID

|              | gFID SiT-B w/o cfg | gFID SiT-XL w/o cfg | iFID |
|--------------|----------------|---------------|------|
| SD-VAE       |                |               |      |
| FLUX-VAE     |                |               |      |
| QW-VAE       |                |               |      |
| SD3-VAE      |                |               |      |
| EQ-VAE       |                |               |      |
| IN-VAE       |                |               |      |
| VA-VAE       |                |               |      |
| VA-VAE (c64) |                |               |      |
| SOFT-VQ      |                |               |      |
| MAE-TOK      |                |               |      |
| DE-TOK       |                |               |      |
| DM-VAE       |                |               |      |
| REPAE-VAE    |                |               |      |
| RAE          |                |               |      |

## To Include Your VAE
* implement your vae in a separate py fite in ./ifid/vae/
* add config file in ./configs/
* submit a pull request
