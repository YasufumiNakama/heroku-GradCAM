from GradCAM import gradcam, model, transforms
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImageDataset(Dataset):
    def __init__(self, image_dir, file_paths, transform=None):
        self.image_dir = image_dir
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = f'{self.image_dir}/{self.file_paths[idx]}.jpg'

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image


def test_plot():
    dataset = ImageDataset(image_dir='./img', file_paths=['bee'], transform=transforms.get_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    gradcam.plotGradCAM(model.model, model.final_conv, model.fc_params, loader, device=device)


if __name__ == '__main__':
    test_plot()
