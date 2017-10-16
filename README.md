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

## Accuracy
| Model | Implementation |   epoch |   iteration | Accuracy | Accuracy Class | Mean IU | FWAV Accuracy |
|:-----:|:--------------:|:-------:|:-----------:|:--------:|:--------------:|:-------:|:-------------:|
|FCN32s      | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn32s)       | - | -     | 90.49 | 76.48 | 63.63 | 83.47 |
|FCN32s| Ours|8 | 68000 | 90.31 | 72.36 | 62.37 | 82.83 |
|FCN16s      | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn16s)       | - | -     | 91.00 | 78.07 | 65.01 | 84.27 |
|FCN16s| Ours|2 | 18000 | 90.30| 74.46 | 62.72 | 82.95 |


# Requirements
* pytorch >= 0.2.0
* python 3.6
* CUDA 8.0
* torchvision >= 0.1.8
* visdom >=1.0.1 (for loss and results visualization)
* Pillow for simple image operations
* tqdm
* scipy



# One-line installation
```bash
1. git clone https://github.com/jjcao/segPytorch.git

2. cd segPytorch

3. install pytorch 
conda install pytorch torchvision cuda80 -c soumith

4. pip install -r requirements.txt

[conda install pandas]
[conda install seaborn]
```

# Reference code
1. https://github.com/meetshah1995/pytorch-semseg
2. https://github.com/wkentaro/pytorch-fcn
