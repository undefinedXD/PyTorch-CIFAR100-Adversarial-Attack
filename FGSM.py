#import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.nn as nn
from advertorch.utils import predict_from_logits
from utils import get_test_dataloader
# from advertorch_examples.utils import _imshow
from models.vgg import vgg19_bn
from conf import settings
from advertorch.attacks import GradientSignAttack
from torch.autograd import Variable
from torchvision import utils as vutils
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 3)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor.squeeze()
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    im = Image.fromarray(input_tensor)
    im.save(filename)

torch.manual_seed(42)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = vgg19_bn()
model.load_state_dict(
    torch.load("checkpoint/vgg_baseline.pth"))
model.to(device)
model.eval()

cifar100_test_loader = get_test_dataloader(
    settings.CIFAR100_TRAIN_MEAN,
    settings.CIFAR100_TRAIN_STD,
    #settings.CIFAR100_PATH,
    num_workers=2,
    batch_size=16,
    shuffle=True
)

adversary = GradientSignAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
    clip_min=0.0, clip_max=1.0,
    targeted=False)
correct_1 = 0.0
correct_5 = 0.0
attack_correct_1 = 0.0
attack_correct_5 = 0.0
total = 0
for n_iter, (image, label) in enumerate(cifar100_test_loader):
    print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
    image = Variable(image).cuda()
    label = Variable(label).cuda()
    output = model(image)
    adv_untargeted = adversary.perturb(image,None)
    attack_output = model(adv_untargeted)
    # for i in range(16):
    #     save_image_tensor2pillow(image[i],str(i)+'.jpg')
    #     save_image_tensor2pillow(adv_untargeted[i],'attack_'+str(i)+'.jpg')    
    _, pred = output.topk(5, 1, largest=True, sorted=True)
    _attack,pred_attack = attack_output.topk(5, 1, largest=True, sorted=True)
    # pred为output的class index
    label = label.view(label.size(0), -1).expand_as(pred)
    correct = pred.eq(label).float()
    attack_correct = pred_attack.eq(label).float()
    print(pred)
    print("-"*50)
    print(pred_attack)
    #compute top 5
    correct_5 += correct[:, :5].sum()
    #compute top1 
    correct_1 += correct[:, :1].sum()
    attack_correct_5 += attack_correct[:, :5].sum()
    attack_correct_1 += attack_correct[:, :1].sum()

print()
#Top-1 error = （正确标记 与 模型输出的最佳标记不同的样本数）/ 总样本数；
#Top-5 error = （正确标记 不在 模型输出的前5个最佳标记中的样本数）/ 总样本数；
print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))
print("Attack Top 1 err",1 - attack_correct_1 / len(cifar100_test_loader.dataset))
print("Attack Top 5 err",1 - attack_correct_5 / len(cifar100_test_loader.dataset))