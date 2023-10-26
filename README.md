## Generative Models in Torch

This repository is a collection of generative models for scenes in PyTorch. The following models have been implemented-
* Variational Autoencoders (VAEs) [`vae.py`](./algorithms/vae.py)
* Generative Adversarial Networks (GANs) [`gan.py`](./algorithms/gan.py)
* Energy-based Model (EBM) [`ebm.py`](./algorithms/ebm.py)
* Variational Energy-based Model (VAEBM) [`vaebm.py`](./algorithms/vaebm.py)
* Energy-based Autoencoder (EAE) [`ebmvae_1.py`](./algorithms/ebmvae_v1.py)
* Score-based Model/Noise Contrastive Score Network (NCSN) [`ncsn.py`](./algorithms/ncsn.py)

## Usage

Download the manipulation scenes dataset from [this link](https://drive.google.com/drive/folders/1jxBQE1adsFT1sWsfatbhiZG6Zkf3EW0Q?usp=sharing) and place it in the `data` directory.

Train a model by running its respective file. For instance, train the VAEBM using the following-

```
python train_vaebm.py
```

## Citation

If you find this repository helpful then please cite the following-
```
@misc{ebmrl,
  author = {Suri, Karush},
  title = {{Generative Models in Torch}},
  url = {https://github.com/karush17/ebm-rl},
  year = {2021}
}
```
