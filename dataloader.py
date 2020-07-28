from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
import os.path as osp
import numpy as np
from PIL import Image

from utils import Config

class kitti_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.train_dir = Config['train_path']
        self.test_dir = Config['test_path']
        self.transforms = self.get_data_transforms()
        self.X_train_dir = osp.join(self.train_dir, 'image_2')
        self.y_train_dir = osp.join(self.train_dir, 'gt_image_2')

    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop((370,1240)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop((370,1240)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms

    # read train data files
    def create_trainset(self):
        X, y = [], []
        X_files = sorted([osp.join(self.X_train_dir, file) for file in os.listdir(self.X_train_dir)])
        y_files = sorted([osp.join(self.y_train_dir, file) for file in os.listdir(self.y_train_dir)])
        for x_item, y_item in zip(X_files, y_files):
            X.append(x_item)
            y.append(y_item)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
        print('X_train:', np.shape(X_train))
        print('X_val:', np.shape(X_val))
        print('y_train:', np.shape(y_train))
        print('y_val:', np.shape(y_val))
        return X_train, X_val, y_train, y_val


class kitti_train(Dataset):
    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        return self.transform(Image.open(self.X_train[item])), self.transform(Image.open(self.y_train[item]))


class kitti_val(Dataset):
    def __init__(self, X_val, y_val, transform):
        self.X_val = X_val
        self.y_val = y_val
        self.transform = transform

    def __len__(self):
        return len(self.X_val)

    def __getitem__(self, item):
        return self.transform(Image.open(X_val[item])), self.transform(Image.open(y_val[item]))


def get_train_dataloader(debug, batch_size, num_workers):
    dataset = kitti_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_val, y_train, y_val = dataset.create_trainset()

    if debug == True:
        train_set = kitti_train(X_train[:20], y_train[:20], transform=transforms['train'])
        val_set = kitti_val(X_val[:10], y_val[:10], transform=transforms['val'])
        dataset_size = {'train': train_set.__len__(), 'val': val_set.__len__()}
    else:
        train_set = kitti_train(X_train, y_train, transforms['train'])
        val_set = kitti_val(X_val, y_val, transforms['val'])
        dataset_size = {'train': train_set.__len__(), 'val': val_set.__len__()}

    print(dataset_size)

    datasets = {'train': train_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x == 'train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                   for x in ['train', 'val']}
    return dataloaders, dataset_size



#if __name__=="__main__":
#    get_train_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])