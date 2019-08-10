from GradCAM import gradcam, model, transforms
import torch
from torch.utils.data import DataLoader
from GradCAM.dataset import ImageDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_plot():
    dataset = ImageDataset(image_dir='./img', file_paths=['bee.jpg'], transform=transforms.get_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    gradcam.plotGradCAM(model.model, model.final_conv, model.fc_params, loader, device=device)


if __name__ == '__main__':
    test_plot()
