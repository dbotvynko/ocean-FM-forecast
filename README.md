# ocean-DDPMs


## Install

```
git clone https://github.com/pierreHaslee/ocean-DDPMs.git
cd ocean-DDPMs
conda install -c conda-forge mamba
conda create -n ddpm-env
conda activate ddpm-env
mamba env update -f environment.yaml
pip install ocean4dvarnet
```

The DDPM Unet model originates from:
https://github.com/mattroz/diffusion-ddpm/tree/main