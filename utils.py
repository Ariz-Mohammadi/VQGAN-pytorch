import os
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size
        self.images = []
        
        # Recursively find all image files in the dataset directory
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    self.images.append(os.path.join(root, file))
        
        self._length = len(self.images)
        
        # Define transformations: resize and crop
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        
        # Ensure the image is in grayscale mode
        if image.mode != "L":
            raise ValueError(f"Expected grayscale image but got {image.mode}.")
        
        # Convert to numpy array (grayscale, single-channel)
        image = np.array(image).astype(np.uint8)  # Shape [H, W]
        
        # Add a channel dimension for preprocessing ([H, W] -> [H, W, 1])
        image = image[:, :, None]
        
        # Apply Albumentations preprocessing
        image = self.preprocessor(image=image)["image"]  # Shape [H, W, 1] -> [H, W, 1]
        
        # Remove the extra channel dimension ([H, W, 1] -> [H, W])
        image = np.squeeze(image, axis=-1)
        
        # Normalize and add channel dimension for PyTorch ([H, W] -> [1, H, W])
        image = (image / 127.5 - 1.0).astype(np.float32)  # Normalize to [-1, 1]
        image = image[None, :, :]  # Add channel dimension
        return image


    def __getitem__(self, index):
        image_path = self.images[index]
        image = self.preprocess_image(image_path)
        return image

def load_data(args):
    train_data = ImagePaths(args.dataset_path, size=256)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    return train_loader


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()
