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
        "ifid.vae.rae_module",
        "ifid.vae.rae_module.encoders",
        "ifid.vae.rae_module.decoders",
    ],
)
