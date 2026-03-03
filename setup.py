from setuptools import setup

setup(
    name="ifid",
    version="0.0.0",
    packages=[
        "ifid",
        "ifid.sit",
        "ifid.fid",
        "ifid.vae",
        "ifid.vae.continous_tokenizer",
        "ifid.vae.continous_tokenizer.modules",
        "ifid.vae.continous_tokenizer.modules.timm_vit",
        "ifid.vae.continous_tokenizer.quantizers",
        "ifid.vae.rae_module",
        "ifid.vae.rae_module.encoders",
        "ifid.vae.rae_module.decoders",
    ],
)
