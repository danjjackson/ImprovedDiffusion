import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def show_images(data, num_samples=20, cols=4):

    transform = transforms.Compose(
        [transforms.PILToTensor(),
         transforms.Resize((IMG_SIZE, IMG_SIZE))
            ]
        )
    images = [transform(data[i][0]) for i in range(20)]
    image_grid = torchvision.utils.make_grid(images, nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0))

def load_transformed_dataset(img_size):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(
        root=".",
        download=True,
        transform=data_transform
    )
    test = torchvision.datasets.StanfordCars(
        root=".",
        download=True,
        transform=data_transform,
        split='test'
    )
    return torch.utils.data.ConcatDataset([train, test])

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]

    image = reverse_transforms(image)
    plt.imshow(image)
    #image.save('test_noised_image.jpg')

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def create_dataloader(batch_size, img_size):
    data = load_transformed_dataset(img_size)
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    return dataloader