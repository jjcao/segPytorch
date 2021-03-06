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
|FCN32s| Ours|11 | 95000 | 89.59| 75.11 | 62.56 | 81.93 |
|FCN32s| Ours|19 | 173000 | 89.82 | 74.30 | 62.86 | 82.19 |
|FCN16s      | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn16s)       | - | -     | 91.00 | 78.07 | 65.01 | 84.27 |
|FCN16s| Ours|22 | 190000 | 90.05| 77.02| 63.91 | 82.70 |
|FCN8s       | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s)        | - | -     | 91.23 | 77.61 | 65.51 | 84.55 |
|FCN8s| Ours | 11 | 100000 | 90.25 | 77.55 | 64.63 | 83.02 |

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

### Usage

**To train the model :**

```
python train.py -c configuration_file

configuration_file: such as config_fcn32s.ini
```

**To validate the model :**

```
python validate.py -m MODEL_PATH [-d DATASET_PATH] [-s SPLIT] [-g 1]

  -m   Path to the saved model, such as '../../output/fcn32s_model_best.pth.tar'
  -d   root path of dataset, such as '../../data/
  -s    Split of dataset to validate on, such as VOC2011, then ../../data/VOC/VOCdevkit/VOC2012 will be used
  -g   gpu or cpu, -1 for cpu
```

**To test the model w.r.t. a dataset on custom images(s):**

```
python test.py -m MODEL_PATH -i IMG_PATH [-o OUT_PATH]
```

# Reference code
1. https://github.com/meetshah1995/pytorch-semseg
2. https://github.com/wkentaro/pytorch-fcn
