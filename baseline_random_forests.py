# Preprocessing images in our dataset
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import os

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import seaborn as sns

from plot_report import plot_classification_report

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
            # place rgb channel at end 
            image = np.transpose(image, (1, 2, 0))
        
        #img_one_hot = torch.zeros(15)
        #img_one_hot[img_class] = 1
        sample = {'image': image, 'img_class': img_class}
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

def data_loader(shuffle=True, num_workers=0):
    """
    Returns DataLoader objects for train and validation data
    """
    train_images, val_images = filename_loader()

    # normalize the pixel values to between 0 and 1 and crop to same size for DataLoader to work
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])
    img_dir = os.path.join(os.path.dirname(__file__), 'Human Action Recognition', 'train')

    val_dataset = HARDataset(data=val_images, img_dir=img_dir, transform=transform)
    train_dataset = HARDataset(data=train_images, img_dir=img_dir, transform=transform)

    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=shuffle, num_workers=num_workers)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=shuffle, num_workers=num_workers)
    
    return train_dataset, val_dataset

train_set, val_set = data_loader()

x_train = []
y_train = []

x_val = []
y_val = []

for i in range (len(train_set)):
  print(i)
  train_set_img = train_set[i]['image'].numpy()
  x_train.append(train_set_img.flatten())
  y_train.append(train_set[i]['img_class'])

for i in range(len(val_set)):
  print(i)
  val_set_img = val_set[i]['image'].numpy()
  x_val.append(val_set_img.flatten())
  y_val.append(val_set[i]['img_class'])

x_train_array = np.array(x_train)
y_train_array = np.array(y_train)

x_val_array = np.array(x_val)
y_val_array = np.array(y_val)

assert len(x_train_array) == len(y_train_array)
p = np.random.permutation(len(x_train_array))
x_train_array, y_train_array = x_train_array[p], y_train_array[p]

model=RandomForestClassifier(n_estimators=250)
model.fit(x_train_array,y_train_array)

y_pred=model.predict(x_val_array)
accuracy_score(y_pred,y_val_array)
print(classification_report(y_pred,y_val_array))
cf_matrix = confusion_matrix(y_pred,y_val_array)
print(cf_matrix)

plt.figure(figsize=(10,7))
sns.heatmap(cf_matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('confusion_matrix.png')

#plot_classification_report(report)
#plt.savefig('baseline_plot_classif_report.png', dpi=200, format='png', bbox_inches='tight')
#plt.close()

