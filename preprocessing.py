# Preprocessing images in our dataset
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import os
import torchvision
from tqdm import tqdm

alexnet = torchvision.models.alexnet(pretrained=True)

class HARDataset(Dataset):
    """
    Custom dataset for HAR images that returns transformed image array and corresponding class
    """
    
    def __init__(self, data, img_dir, transform=None):
        """
        Inputs:
            data (list): 2D list in the form of [[class, file_name]]
            img_dir (String): directory of the folder with all the images
            transform (torchvision.transforms): transformation to be applied to images
        """
        self.data = np.array(data)
        self.img_dir = img_dir
        self.transform = transform
        self.alexnet = alexnet.features.requires_grad_(False)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_name = os.path.join(self.img_dir, self.data[idx, 1])
        img_class = int(self.data[idx, 0])
        image = plt.imread(image_name)
        if self.transform:
            image = self.transform(np.array(image))
        
        img_one_hot = torch.zeros(15)
        img_one_hot[img_class] = 1
        sample = {'image':  self.alexnet(image), 'img_class': img_one_hot}
        return sample

def filename_loader():
    """
    Helper function that loads image file name and corresponding class from 'Human Action Recognition' folder
    """
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Human Action Recognition', 'Training_set.csv'))

    # one-hot encoding not needed for nn.CrossEntropyLoss()(x,y)
    # y is the index for the class (0 to 14) and x is the output from the model without sigmoid (1x15 tensor)
    classes = {
        "sitting":0,
        "using_laptop":1,
        "hugging":2,
        "sleeping":3,
        "drinking":4,
        "clapping":5,
        "dancing":6,
        "cycling":7,
        "calling":8,
        "laughing":9,
        "eating":10,
        "fighting":11,
        "listening_to_music":12,
        "running":13,
        "texting":14,
    }
    images = {
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
        6:[],
        7:[],
        8:[],
        9:[],
        10:[],
        11:[],
        12:[],
        13:[],
        14:[]
    }
    train_images = []
    val_images = []

    for _, data in train.iterrows():
        img_class = classes[data[1]]
        images[img_class].append([img_class, data[0]])
    
    for img_class, imgs in images.items():
        train_split = int(len(imgs) * 0.8) # 80/20 training/validation split for each class
        train_images += imgs[:train_split]
        val_images += imgs[train_split:]

    return train_images, val_images

def data_loader(batch_size=64, shuffle=True, num_workers=0):
    """
    Returns DataLoader objects for train and validation data
    """
    train_images, val_images = filename_loader()

    # normalize the pixel values to between 0 and 1 and crop to same size for DataLoader to work
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])
    img_dir = os.path.join(os.path.dirname(__file__), 'Human Action Recognition', 'train')

    val_dataset = HARDataset(data=val_images, img_dir=img_dir, transform=transform)
    train_dataset = HARDataset(data=train_images, img_dir=img_dir, transform=transform)

    transform_flip = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=1.0), transforms.Resize((224,224))])
    train_dataset_flipped = HARDataset(data=train_images, img_dir=img_dir, transform=transform_flip)

    transform_rotate_90 = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(degrees=(90,90)), transforms.Resize((224,224))])
    train_dataset_rotated90 = HARDataset(data=train_images, img_dir=img_dir, transform=transform_rotate_90)

    transform_rotate_180 = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(degrees=(180,180)), transforms.Resize((224,224))])
    train_dataset_rotated180 = HARDataset(data=train_images, img_dir=img_dir, transform=transform_rotate_180)

    transform_rotate_270 = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(degrees=(270,270)), transforms.Resize((224,224))])
    train_dataset_rotated270 = HARDataset(data=train_images, img_dir=img_dir, transform=transform_rotate_270)

    transform_random_crop = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()])
    
    train_dataset_cropped = HARDataset(data=train_images, img_dir=img_dir, transform=transform_random_crop)

    # Apply Gaussian Noise

    class gaussian_noise(object): 
        def __call__(self, image):
            return image + torch.randn_like(image) * 0.1


        def __repr__(self):
            return self.__class__.__name__+'()'

    
    transform_gaussian_noise = transforms.Compose([transforms.ToTensor(), gaussian_noise(), transforms.Resize((224,224))])
    train_dataset_noise = HARDataset(data=train_images, img_dir=img_dir, transform=transform_gaussian_noise)

    train_dataset = ConcatDataset([train_dataset, train_dataset_flipped, train_dataset_rotated90, train_dataset_rotated180, train_dataset_rotated270, train_dataset_cropped, train_dataset_noise])

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader, val_loader

'''
Create a new, complete dataset with all images (augmented data included) embedded with AlexNet weights in the form of np arrays.
Using this new dataset for future training (with a new dataloader) should be significantly less time consuming compared to the 
original method. 

'''

def new_dataset(batch_size=1, dire = "alex_embedding_set"):
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(0)

    train_loader, val_loader = data_loader(batch_size=batch_size, shuffle=False)
    id=0
    for batch in tqdm(train_loader):
        imgs, labels = batch.values()
        labels = torch.argmax(labels, dim=1)
        np.save(f"{dire}/embed_{id}", imgs.numpy())
        np.save(f"{dire}/label_{id}", labels.numpy())
        id += 1
    
    for batch in tqdm(val_loader):
        imgs, labels = batch.values()
        labels = torch.argmax(labels, dim=1)
        np.save(f"{dire}/embed_{id}", imgs.numpy())
        np.save(f"{dire}/label_{id}", labels.numpy())
        id += 1

def test_filename_loader():
    """
    Helper function that loads image file name and corresponding class from 'Human Action Recognition' folder
    """
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Human Action Recognition', 'Testing_set.csv'))

    # one-hot encoding not needed for nn.CrossEntropyLoss()(x,y)
    # y is the index for the class (0 to 14) and x is the output from the model without sigmoid (1x15 tensor)
    classes = {
        "sitting":0,
        "using_laptop":1,
        "hugging":2,
        "sleeping":3,
        "drinking":4,
        "clapping":5,
        "dancing":6,
        "cycling":7,
        "calling":8,
        "laughing":9,
        "eating":10,
        "fighting":11,
        "listening_to_music":12,
        "running":13,
        "texting":14,
    }
    images = {
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
        6:[],
        7:[],
        8:[],
        9:[],
        10:[],
        11:[],
        12:[],
        13:[],
        14:[]
    }
    test_images = []

    for _, data in test.iterrows():
        if (data[1] in classes.keys()):
            img_class = classes[data[1]]
            images[img_class].append([img_class, data[0]])
    
    for imgs in images.values():
        test_images += imgs

    return test_images

def test_data_loader(batch_size=64, shuffle=True, num_workers=0):
    """
    Returns DataLoader objects for train and validation data
    """
    test_images = test_filename_loader()

    # normalize the pixel values to between 0 and 1 and crop to same size for DataLoader to work
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.FiveCrop(150), transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])), transforms.Resize((224, 224))])
    img_dir = os.path.join(os.path.dirname(__file__), 'Human Action Recognition', 'test')

    test_dataset = HARDataset(data=test_images, img_dir=img_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return test_loader


class AlexEmbed(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, length = 70560, root_dir='', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.length = length
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = np.load(self.root_dir+f"alex_embedding_set/embed_{idx}.npy")
        label = np.load(self.root_dir+f"alex_embedding_set/label_{idx}.npy")

        sample = {'image': image, 'label': np.squeeze(label)}

        if self.transform:
            sample = self.transform(sample)

        return sample

class AlexEmbed_test(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, length = 207, root_dir='', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.length = length
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        cur_dir = os.getcwd()
        image = np.load(cur_dir+f"/testset/embed_{idx}.npy")
        label = np.load(cur_dir+f"/testset/label_{idx}.npy")
        
        sample = {'image': image, 'label': np.squeeze(label)}

        if self.transform:
            sample = self.transform(sample)

        return sample