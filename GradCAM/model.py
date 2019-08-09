from torchvision import models
from torch.nn import functional as F
from torch import nn


model = models.resnet18(pretrained=True)


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


model.avgpool = AvgPool()

"""
print(model)
for name, module in model._modules.items():
    print(name)
"""

final_conv = model._modules.get('layer4')[1]._modules.get('conv2')
fc_params = list(model._modules.get('fc').parameters())
