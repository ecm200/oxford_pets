import os

import pandas as pd
from torch.utils.data import Dataset
#from torchvision.io import read_image
from torch.utils.data import DataLoader
from PIL import Image

class OxfordPetsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, skiprows=0, img_file_ext='.jpg'):
        self.img_file_ext = img_file_ext
        self.img_labels = pd.read_csv(annotations_file, header=None, skiprows=skiprows, delim_whitespace=' ', names=['Class Name ID','Class ID','Species ID','Breed ID'])
        self.img_labels['Class Name'] = self.img_labels['Class Name ID'].str.rsplit('_', n=1, expand=True)[0]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(self.img_labels['Class Name'].unique())
        self.imgs = self.img_labels['Class Name ID']+img_file_ext

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels['Class Name ID'].iloc[idx]+self.img_file_ext)
        #image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels['Class Name'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def create_dataloaders(data_transforms, data_dir, batch_size, num_workers, train_file, test_file, shuffle=None, test_batch_size=2):
    '''
    Given directories of images for train and test datasets, organized in sub folders of class under train and test directories, build dataloader objects to serve to the network.

    Cub Tools
    Ed Morris (c) 2021
    '''


    # Set the name of the train and test image directories
    images_lists = {'train' : train_file, 'test' : test_file}

    # Set the batch sizes for each operation
    batch_size = {'train' : batch_size, 'test' : test_batch_size}

    # Set the shuffle option dict for each operation
    if shuffle == None:
        shuffle = {'train' : True, 'test' : False}

    # Setup data loaders with augmentation transforms
    
    image_datasets = {x: OxfordPetsDataset(annotations_file=images_lists[x], img_dir=data_dir, transform=data_transforms[x])
                    for x in ['train', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size[x],
                                 shuffle=shuffle[x], num_workers=num_workers)
                for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    print('***********************************************')
    print('**            DATASET SUMMARY                **')
    print('***********************************************')
    for dataset in dataset_sizes.keys():
        print(dataset,' size:: ', dataset_sizes[dataset],' images')
    print('Number of classes:: ', len(class_names))
    print('***********************************************')
    print('[INFO] Created data loaders.')

    return dataloaders['train'], dataloaders['test']