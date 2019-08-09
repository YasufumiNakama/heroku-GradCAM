from torchvision.transforms import Compose, ToTensor, Normalize, Resize

IMG_SIZE = 128

get_transform = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
