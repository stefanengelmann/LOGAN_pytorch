# PyTorch implementation of LOGAN
This repository is a PyTorch implementation of the [LOGAN: Unpaired Shape Transform in Latent Overcomplete Space](https://arxiv.org/pdf/1903.10170.pdf) paper

# Setup
```shell
conda create -n logan_pytorch python=3.11
pip install -r requirements
```
This implementation uses CUDA optimized implementation of the [PointNet++ Set Abstraction layers](https://github.com/erikwijmans/Pointnet2_PyTorch) and the [EMD loss](https://github.com/daerduoCarey/PyTorchEMD) which both requires code to be compiled with the NVIDIA CUDA Compiler Driver (NVCC)
Check if you have NVCC installed:
```shell
nvcc -V
```
If not, check your CUDA version with
```shell
nvidia-smi
```
and install the relevant version with conda (here shown with 12.2)
```shell
conda install cuda -c nvidia/label/cuda-12.2.0
```
Finally, build the CUDA kernels
```shell
pip install losses/PyTorchEMD/.
pip install pointnet2_ops_lib/.
```

# Training
Training LOGAN is a two-step process. First the autoencoder, then the translator
```shell
python train_ae.py --root_dir=path/to/data/ --save_best
python train_translator --root_dir=path/to/data/ --save_models
```

# Evaluation
Evaluating the autoencoder can be done with either EMD or Chamfer loss 
```shell
python test_ae.py --root_dir=path/to/data/ --ae_loss=emd
python test_ae.py --root_dir=path/to/data/ --ae_loss=chamfer
```
Visual results can also be obtained by setting the `--visualize` flag. An HTML file with an interactive [K3D](https://github.com/K3D-tools/K3D-jupyter) 3D plot will be generated and saved to the project root folder and can then easily be opened in any browser

Inference can be performed visually with K3D
```shell
python inference.py --root_dir=path/to/data/ 
```
Screenshot here

# Data
Data can be found on the original [LOGAN](https://github.com/kangxue/LOGAN) repository page

# Acknowledgements
- Original LOGAN paper repo: https://github.com/kangxue/LOGAN
- PointNet++ Set Abstraction layers: https://github.com/erikwijmans/Pointnet2_PyTorch 
- EMD loss: https://github.com/daerduoCarey/PyTorchEMD
- Some code including the Chamfer loss, data loader and gradient penalty is adopted from https://github.com/Yunyung/Characteristic-preserving-Latent-Space-for-Unpaired-Cross-domain-Translation-of-3D-Point-Clouds

