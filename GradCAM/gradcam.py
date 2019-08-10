import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from . import transforms, classes

IMG_SIZE = transforms.IMG_SIZE
class_dict = classes.classes


class SaveFeatures:
    """ Extract pretrained activations"""
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()

    def remove(self):
        self.hook.remove()


def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv[0, :, :, ].reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img


def plotGradCAM(model, final_conv, fc_params, loader, img_size=IMG_SIZE, device='cpu'):
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    # save activated_features from conv
    activated_features = SaveFeatures(final_conv)
    # save weight from fc
    weight = np.squeeze(fc_params[0].cpu().data.numpy())
    # heatmap images
    fig, ax = plt.subplots()
    for i, img in enumerate(loader):
        output = model(img.to(device))
        output = F.softmax(output.data.squeeze(), dim=0)
        pred_idx = np.argmax(output.cpu().detach().numpy())
        cur_images = img.cpu().numpy().transpose((0, 2, 3, 1))
        heatmap = getCAM(activated_features.features, weight, pred_idx)
        plt.imshow(cv2.cvtColor(cur_images[0], cv2.COLOR_BGR2RGB))
        plt.imshow(cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_LINEAR), alpha=0.4, cmap='jet')
        ax.set_title('Predict: %s' % class_dict[pred_idx], fontsize=14)
        break
    plt.show()
    return fig
