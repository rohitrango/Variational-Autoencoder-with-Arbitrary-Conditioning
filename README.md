# Variational Autoencoder with Arbitrary Conditioning

## Introduction
This is a PyTorch implementation of the ICLR 2019 paper [Variational Autoencoder with Arbitrary Conditioning](https://openreview.net/forum?id=SyxtJh0qYm). The idea of the paper is to extend the notion of Conditional Variational Autoencoders to enable arbitrary conditioning, i.e. a different subset of features is used to generate samples from the distribution <a href="https://www.codecogs.com/eqnedit.php?latex=p(x_b&space;|&space;x_{1-b},&space;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x_b&space;|&space;x_{1-b},&space;b)" title="p(x_b |x_{1-b}, b)" /></a> where <a href="https://www.codecogs.com/eqnedit.php?latex=x_{1-b}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{1-b}" title="x_{1-b}" /></a> is a random subset of observed features and _b_ is a binary mask indicating if a feature is observed or not. The authors postulate that this can be effective especially for imputation and inpainting tasks where only a certain portion of the image is visible.


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

### Baselines
The metric used by the paper is negative log-likelihood for a bernoulli distribution (since the digits are binarized). Although the log-likelihood is not expected to be very low since there can be many probable images with the same value of observed image, and hence a single solution may not be the image corresponding to the ground truth but may still be feasible. However, the metric is still well-behaved.

|		| Negative log-likelihood | 
|-------|-------------------------|
|MNIST	| 0.18557				  |

The results are a little more useful for the image imputation tasks since the probability space of feasible images is now restricted given the image and mask. These are some of the baseline tasks to check sanity of the code and method. The paper measures PSNR scores between their and other methods. Note that these numbers are just some baselines used to perform a sanity check.


| PSNR   | Random mask (p = 0.9)	 | Square mask (W = 20)  |
|--------|---------------------------|-----------------------|
| CelebA | 		22.1513				 | 		28.7348			 |	  

The other results show numbers for the specific task given for CelebA dataset.

### Inpainting tasks
Here is the table for PSNR of inpaintings for different masks. Higher values are better. I used a bigger architecture and ran for more iterations since I didn't have the exact architecture and training iterations in the paper. Here are some results:

| Method/Masks        			 | Center 		| Pattern 	| Random 	| Half  	|
|--------------------------------|--------------|-----------|-----------|-----------|
| Context Encoder<sup>*</sup>    | 21.3   		| 19.2    	| 20.6   	| 15.5  	|
| SIIDGM<sup>*</sup>  	      	 | 19.4   		| 17.4    	| 22.8   	| 13.7  	|
| VAEAC (1 sample)<sup>*</sup>   | 22.1  		| 21.4    	| **29.3**  | 14.9  	|
| VAEAC (10 samples)<sup>*</sup> | 23.7   		| 23.3    	| **29.3**  | 17.4  	|
| VAEAC (this repo)   			 | **25.02**  	| **24.60** | 24.93  	| **17.48** |

<sup>*</sup> = values taken from the paper


Here is the table for PSNR for inpainting using other masks. Higher values are better. Again, my values of PSNR are much more than reported in the paper. It could be because I'm using a bigger architecture and training for longer.

| Method/Masks        			 | O1 			| O2 		| O3 		| O4	  	| O5	  	| O6 		|
|--------------------------------|--------------|-----------|-----------|-----------|-----------|-----------|
| Context Encoder<sup>*</sup>    | 18.6   		| 18.4    	| 17.9  	| 19.0  	| 19.1		| 19.3		|
| GFC<sup>*</sup>  	      	 	 | 20.0   		| 19.8    	| 18.8  	| 19.7  	| 19.5		| 20.2		|
| VAEAC (1 sample)<sup>*</sup>   | 20.8  		| 21.0    	| 19.5  	| 20.3  	| 20.3		| 21.0		|
| VAEAC (10 samples)<sup>*</sup> | 22.0   		| 22.2    	| 20.8  	| 21.7  	| 21.8 		| 22.2		|
| VAEAC (this repo)   			 | **27.59**	| **33.20** | **30.32** | **31.38** | **32.28** | **28.20** |


## Qualitative Results
Here are some qualitative results on MNIST. The first image is the input to VAEAC. The other images are the samples drawn from the network. The last image is the ground truth. Note that given the partially observed input, the network successfully learns to output feasible outputs. The conditioning is arbitrary because each instance has a different subset of pixels which is observed. 

![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/0.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/1.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/2.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/3.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/4.png)
![mnist-test](https://github.com/rohitrango/ICLR-challenge/blob/master/images/MNIST/5.png)

Here are some results on CelebA dataset with only 10% pixels retained in the input.

![celeba-baseline](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random_baseline/0.png)
![celeba-baseline](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random_baseline/1.png)
![celeba-baseline](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random_baseline/2.png)
![celeba-baseline](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random_baseline/3.png)
![celeba-baseline](https://github.com/rohitrango/ICLR-challenge/blob/master/images/celebA_random_baseline/8.png)

Check out [`RESULTS.md`](https://github.com/rohitrango/ICLR-challenge/blob/master/RESULTS.md) for more results.

## Updates
**17 Mar 2019:** Set up the directory structure and general flow of the paper after reading and understanding the paper at the implementation level. Added dataloaders for MNIST and CelebA.

**19 Mar 2019:** Compiled results for MNIST. Qualitative results look good, although not as diverse as the paper showed. Quantitative results are in the table. Also added results for CelebA dataset, with box masks, and dropping of random pixels.

**20 Mar 2019:** Starting actual experiments performed in the paper now that sanity check is done.

**21 Mar 2019:** Ran experiments on three tasks for CelebA - *Center, Random*, and *Half*. Both qualitative and quantitative results are at par with the paper. 

**22 Mar 2019:** Ran experiments on all 4 tasks and added qualitative and quantitative results. Also wrote the code for the second set of experiments with O1-O6 masks. Quantitative results look good.
