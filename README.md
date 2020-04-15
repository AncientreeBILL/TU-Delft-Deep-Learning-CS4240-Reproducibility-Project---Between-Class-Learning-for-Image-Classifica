# Reproducibility project of Between Class Learning for Image Classification

## At the beginning
This project is meant to be a part of the reproducibility project for a master course Deep Learning (CS4240) in TU Delft. We tried to reproduce the result of Between-class Learning for Image Classification by Tokozume et al. 
Here is the original links: 
github: https://github.com/mil-tokyo/bc_learning_image and paper: https://arxiv.org/abs/1711.10284. If you would like to learn more about this topic, please move to the author's webpage.

## About the code
In this reporducibility project, we did our main part on Google colab. The whole package was uploaded as a file on a personal Google Drive.

To try our code, I suggest to run the jupyter notebook file on Google colab. Thus, you also need to upload the code files on your Google Drive. Because we faced some troubles about the file system, Cupy and Chainer. We eventually managed to solve them all but the solution was only tested on Colab. 

To start with, please take a look at your chainer version. We alter the version of chainer from v1.24(the author used) to v3(which is default for colab). Our setup of Cuda and Cupy is also for the Ubuntu virtual machine provided by colab. The most important thing is that we used the GPU mode for our jupyter notebook on colab. We've never tried if it runs well also locally.

## More Imformation
More information of our project is on the blog.
