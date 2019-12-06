# Pytorch_CIFAR100&Adversarial Attack
This is my MachineLearing coursework and My work is Based On This [Repo](https://github.com/weiaicunzai/pytorch-cifar100)

## Main Work

- [x] Use ResNet50、MobileNet、VGG19 as Backbone Network Respectively
- [x] Use Tricks such as Xavier Init、LabelSmoothing ON VGG19
- [x] Use FGSM and PGD algorithms to attack VGG19 model

# Classification Task Results
|network|params|top1 err|top5 err|total epoch|
|:---:|:---:|:---:|:---:|:---:|
|MobileNet|3.3M|32.50|9.67|160|
|VGG19|39.0M|28.82|10.58|160|
|ResNet50|23.7M|22.33|5.50|160|

- Based On VGG19,I also did some experiments.

|      Tricks       | top1 err | top5 err | total epoch | ACC            |
| :---------------: | :------: | :------: | :---------: | -------------- |
|     Original      |  28.82   |  10.58   |     160     | 71.18%         |
| WarmUp&XavierInit |  27.71   |  10.49   |     160     | 72.29%         |
|   No Bias Decay   |  29.86   |  11.96   |     160     | drops about 1% |
|  Label Smoothing  |  26.44   |  10.92   |     160     | 73.56%         |

# Adversairal Attack Results

There is some error in my code when using **FGSM** algorithm, cuz if you settle eps=0,you can still get a good attack effect(acc from nearly 72% to 26%).~~Maybe fix in some weeks~~

| eps  | top1 err | top5 err | ACC    |
| :--: | :------: | :------: | ------ |
|  /   |  28.82   |  10.58   | 71.18% |
|  0   |  74.37   |  57.54   | 25.63% |
| 0.2  |  90.00   |  76.71   | 10.00% |
| 0.3  |  92.28   |  81.06   | 7.72%  |

**PGD results as Following:**（I'm not sure whether its right or not)

| eps  | top1 err | top5 err | ACC    |
| :--: | :------: | :------: | ------ |
|  /   |  28.82   |  10.58   | 71.18% |
|  0   |  28.82   |  10.58   | 71.18% |
| 0.2  |  99.69   |  98.21   | 0.31%  |



# Pytorch-cifar100

practice on cifar100 using pytorch

## Requirements

This is my experiment eviroument, pytorch0.4 should also be fine
- python3.5
- pytorch1.0
- tensorflow1.5(optional)
- cuda8.0
- cudnnv5
- tensorboardX1.6(optional)


## Usage

### 1. enter directory
```bash
$ cd pytorch-cifar100
```

### 2. dataset 
I will use cifar100 dataset from torchvision since it's more convenient, but I also
kept the sample code for writing your own dataset module in dataset folder, as an
example for people don't know how to write it.

### 3. run tensorbard(optional)
Install tensorboardX (a tensorboard wrapper for pytorch)
```bash
$ pip install tensorboardX
$ mkdir runs
Run tensorboard
$ tensorboard --logdir='runs' --port=6006 --host='localhost'
```

### 4. train the model
Train all the model on a Tesla P40(22912MB)   

You need to specify the net you want to train using arg -net

```bash
$ python train.py -net vgg16
```


### 5. test the model
Test the model using test.py
```bash
$ python test.py -net vgg16 -weights path_to_vgg16_weights_file
```

## Implementated NetWork

- vgg [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)
- resnet [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
- mobilenet [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
