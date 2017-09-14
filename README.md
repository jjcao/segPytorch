# segPytorch
My first try.

# Implemented models
FCN32s, pass init test
FCN16s, not tested
FCN8s, not tested
Unet, not tested
Segnet, init test is very bad
Linknet, init testing is as same as segnet. But linknet is very fast.
Pspnet, not tested

# Requirements
* pytorch >= 0.2.0
* python 3.6
* torchvision >= 0.1.8
* visdom >=1.0.1 (for loss and results visualization)
* Pillow for simple image operations
* tqdm
* scipy?



# One-line installation
git clone https://github.com/jjcao/segPytorch.git
cd segPytorch

conda install pytorch cuda80 torchvision -c soumith
`pip install -r requirements.txt`

# Reference code
1. https://github.com/meetshah1995/pytorch-semseg
2. https://github.com/wkentaro/pytorch-fcn
