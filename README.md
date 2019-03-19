# Variational Autoencoder with Arbitrary Conditioning

## Introduction
This is a PyTorch implementation of the ICLR 2019 paper `Variational Autoencoder with Arbitrary Conditioning'.


## Installation
I used Python 2.7 to run and test the code. I recommend using a `virtualenv` environment. Here are the dependencies:
```
opencv-contrib-python==3.4.2.17
scipy>=0.17.0
numpy==1.14.5
torch==0.4.1
torchvision==0.2.1
scikit-image==0.14.2
```
You can use the `requirements.txt` file to install the packages.


## Datasets
- Download the MNIST dataset [here](http://yann.lecun.com/exdb/mnist/) and place it in a directory. Process it using PyTorch's dataloader to get `training.pt` and `test.pt`.

- Download the CelebA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and place it in a directory. I used the aligned and cropped images of size 64x64. The directory structure should be like this:

```
Datasets
├── CelebA
│   └───img_align_celeba
│   	├── 000003.png
│       ├── 000048.png
│       └─── ....
└── MNIST
	└───processed
	    ├── test.pt
	    └── training.pt
```
- For CelebA, the training and validation split is created automatically.


## Training and testing
Training networks from scratch or from pretrained networks is very easy. Simply run:
```
python main.py --mode train --config /path/to/config/file
```
For inference, and visualizations, run the following command
```
python main.py --mode val --config /path/to/config/file
```
The structure of configuration files is inspired from the [Detectron](https://github.com/facebookresearch/Detectron) framework. More details about the configuration files can be found in [`CONFIG.md`](https://github.com/rohitrango/ICLR-challenge/blob/master/CONFIG.md).


## Quantitative Results

### MNIST
The only metric the paper supplied was the negative log-likelihood. Although the log-likelihood doesn't make much sense since there can be many probable images with the same value of observed image, and hence a single solution may not be the image corresponding to the ground truth but may still be feasible.

|		| Negative log-likelihood | 
|-------|-------------------------|
|MNIST	| 0.18557				  |

The results are a little more meaningful for the image imputation tasks since the probability space of feasible images is now restricted given the image and mask. These are some of the baseline tasks to check sanity of the code and method. 

|		 | Random mask (p = 0.9)	 | Square mask (W = 20)  |
|--------|---------------------------|-----------------------|
| CelebA | 		22.1513				 | 		28.7348			 |	  


The other results show numbers for the specific task given for CelebA dataset. (*Coming soon*)


## Qualitative Results
Here are some qualitative results on MNIST. The first image is the input to VAEAC. The other images are the samples drawn from the network. The last image is the ground truth. Note that given the partially observed input, the network successfully learns to output feasible outputs. The conditioning is arbitrary because each instance has a different subset of pixels which is observed. 

![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/0.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/1.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/2.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/3.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/4.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/5.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/6.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/7.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/8.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/9.png)

Here are some results on CelebA dataset with only 10% pixels retained in the input.

![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random/0.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random/1.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random/2.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random/3.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random/8.png)
<!-- ![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random/5.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random/6.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random/7.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random/4.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random/9.png) -->


## Updates
**17 Mar 2019:** Set up the directory structure and general flow of the paper after reading and understanding the paper at the implementation level. Added dataloaders for MNIST and CelebA.

**19 Mar 2019:** Compiled results for MNIST. Qualitative results look good, although not as diverse as the paper showed. Quantitative results are in the table. Also added results for CelebA dataset, with box masks, and dropping of random pixels.

**20 Mar 2019:** Starting actual experiments performed in the paper now that sanity check is done.
