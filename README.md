

# Dual Contrastive Learning Adversarial Generative Networks(DCLGAN)

We provide our PyTorch implementation of unpaired image-to-image translation based on patchwise contrastive learning and dual adversarial learning. DCLGAN (Dual Contrastive Learning Generative Adversarial Networks) is simple yet powerful. Compared to [CycleGAN](https://github.com/junyanz/CycleGAN), DCLGAN performs geometry changes with more realistic results. Compared to [CUT](http://taesung.me/ContrastiveUnpairedTranslation/), DCLGAN is usually more robust and usually achieves better performance. Our model, SimDCL (Similarity DCLGAN) also avoids mode collapse using a new similarity loss. 

DCLGAN is a general model pefroming all kinds of Image-to-Image translation tasks. It achieves SOTA performances in most tasks.

<img src='imgs/dclgan.png' align="right" width=960>
Our pipeline is quite straight forawrd. The main idea is dual setting with two encoders to capture the variability in two distinctive domains.

## Example Results

### Unpaired Image-to-Image Translation
Qualitative results:
<img src="imgs/results.pdf" width="800px"/>

Quantitative results:
<img src="imgs/results2.pdf" width="800px"/>

More visual results:
<img src="imgs/results3.pdf" width="800px"/>



## Prerequisites
see requirements.txt

### Getting started

- Clone this repo:
```bash
git clone
```

- Install PyTorch 1.4 or above and other dependencies (e.g., torchvision, visdom, dominate, gputil).

  For pip users, please type the command `pip install -r requirements.txt`.

  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.


### DCLGAN and SIMDCL Training and Test

- Download the `grumpifycat` dataset (Fig 8 of the paper. Russian Blue -> Grumpy Cats)
```bash
bash ./datasets/download_cut_dataset.sh grumpifycat
```
The dataset is downloaded and unzipped at `./datasets/grumpifycat/`.

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

- Train the DCL model:
```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_DCL 
Or train the SIMDCL model
 ```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_SIMDCL --model simdcl
```
The checkpoints will be stored at `./checkpoints/grumpycat_*/web`.

- Test the DCL model:
```bash
python test.py --dataroot ./datasets/grumpifycat --name grumpycat_DCL
```

The test results will be saved to a html file here: `./results/grumpifycat/latest_test/index.html`.

### DCLGAN, SIMDCL, CUT and CycleGAN
DCLGAN is a more robust unsupervised image-to-image translation model compared to previous models. Our performance is usually better than CUT&CycleGAN.
SIMDCL is a different version, it was designed to solve mode collpase. We recommend using it for small-scale, unbalanced dataset.

### Apply a pre-trained DCL model and evaluate
We provide our pre-trained DCL models for:

Cat <-> Dog :
Horse <-> Zebra:
CityScapes:

### [Datasets](./docs/datasets.md)
Download CUT/CycleGAN/pix2pix datasets and learn how to create your own datasets.

### Citation
If you use our code , please cite our paper.

If you use something included in CUT, please cite [CUT](https://arxiv.org/pdf/2007.15651).
```
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

If you use the original [pix2pix](https://phillipi.github.io/pix2pix/) and [CycleGAN](https://junyanz.github.io/CycleGAN/) model included in this repo, please cite the following papers
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```


### Acknowledgments
Our code is developed based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [CUT](http://taesung.me/ContrastiveUnpairedTranslation/). We thank the awesome work provided by CycleGAN and CUT.
We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation.
