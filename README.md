

# Dual Contrastive Learning Adversarial Generative Networks(DCLGAN)

<img src='imgs/gif_cut.gif' align="right" width=960>

<br><br><br>

<img src='imgs/dclgan.png' align="right" width=960>


We provide our PyTorch implementation of unpaired image-to-image translation based on patchwise contrastive learning and dual adversarial learning. Compared to [CycleGAN](https://github.com/junyanz/CycleGAN), our model trains faster and performs better. Compared to [CUT](http://taesung.me/ContrastiveUnpairedTranslation/), our model is more robust and usually achieves better performance.



We thank Taesung for his work. Our work is inspired by him.
[Contrastive Learning for Unpaired Image-to-Image Translation](http://taesung.me/ContrastiveUnpairedTranslation/)  
 [Taesung Park](https://taesung.me/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/), [Richard Zhang](https://richzhang.github.io/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)<br>
UC Berkeley and Adobe Research<br>
In ECCV 2020


<img src='imgs/patchnce.gif' align="right" width=960>

<br><br><br>

### Pseudo code
```python
import torch
cross_entropy_loss = torch.nn.CrossEntropyLoss()

# Input: f_q (BxCxS) and sampled features from H(G_enc(x))
# Input: f_k (BxCxS) are sampled features from H(G_enc(G(x))
# Input: tau is the temperature used in PatchNCE loss.
# Output: PatchNCE loss
def PatchNCELoss(f_q, f_k, tau=0.07):
    # batch size, channel size, and number of sample locations
    B, C, S = f_q.shape

    # calculate v * v+: BxSx1
    l_pos = (f_k * f_q).sum(dim=1)[:, :, None]

    # calculate v * v-: BxSxS
    l_neg = torch.bmm(f_q.transpose(1, 2), f_k)

    # The diagonal entries are not negatives. Remove them.
    identity_matrix = torch.eye(S)[None, :, :]
    l_neg.masked_fill_(identity_matrix, -float('inf'))

    # calculate logits: (B)x(S)x(S+1)
    logits = torch.cat((l_pos, l_neg), dim=2) / tau

    # return PatchNCE loss
    predictions = logits.flatten(0, 1)
    targets = torch.zeros(B * S, dtype=torch.long)
    return cross_entropy_loss(predictions, targets)
```
## Example Results

### Unpaired Image-to-Image Translation
<img src="imgs/results.gif" width="800px"/>

### Single Image Unpaired Translation
<img src="imgs/singleimage.gif" width="800px"/>


### Russian Blue Cat to Grumpy Cat
<img src="imgs/grumpycat.jpg" width="800px"/>

### Parisian Street to Burano's painted houses
<img src="imgs/paris.jpg" width="800px"/>



## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Update log

04/11/2020: First upload.

### Getting started

- Clone this repo:
```bash
git clone
```

- Install PyTorch 1.1 and other dependencies (e.g., torchvision, visdom, dominate, gputil).

  For pip users, please type the command `pip install -r requirements.txt`.

  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.


### CDCL and SIMDCL Training and Test

- Download the `grumpifycat` dataset (Fig 8 of the paper. Russian Blue -> Grumpy Cats)
```bash
bash ./datasets/download_cut_dataset.sh grumpifycat
```
The dataset is downloaded and unzipped at `./datasets/grumpifycat/`.

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

- Train the DCL model:
```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_DCL --DCL_mode dcl
```
 Or train the SIMDCL model
 ```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_SIMDCL--DCL_mode simdcl
```
The checkpoints will be stored at `./checkpoints/grumpycat_*/web`.

- Test the DCL model:
```bash
python test.py --dataroot ./datasets/grumpifycat --name grumpycat_DCL
```

The test results will be saved to a html file here: `./results/grumpifycat/latest_test/index.html`.

### DCLGAN, SIMDCL, CUT and CycleGAN
DCLGAN is a more robust unsupervised image-to-image translation model compared to previous  models. Our training is faster than CycleGAN and our performance is usually better than CUT&CycleGAN.
SIMDCL is a different version, it was designed to solve mode collpase. We recommend using it for small-scale, unbalanced dataset.

### Apply a pre-trained DCL model and evaluate

We will release all pre-trained models ASAP.

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
Our code is developed based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [CUT](http://taesung.me/ContrastiveUnpairedTranslation/). We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation.
