import numpy as np
import torch
from PIL import Image
from torch.utils import data
import pandas as pd
import os
from torchvision.transforms import v2
'''
to run this file :
 
from load_data import CustomDataLoader
data_loader = CustomDataLoader()
training_generator, test_generator,dev_generator  = data_loader.read_data()

for batch_X, batch_y in training_generator:
    print(batch_X)
    print(batch_y)
    #training processing

'''


OR_PATH = os.getcwd()
os.chdir("..")
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)
FILE_NAME = 'fully_processed.xlsx'
xdf_data = pd.read_excel(FILE_NAME)
xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
xdf_dset_test = xdf_data[xdf_data["split"] == 'test'].copy()
xdf_dset_dev = xdf_data[xdf_data["split"] == 'dev'].copy()
IMAGE_SIZE = 256
CHANNEL = 3
BATCH_SIZE = 80
class CustomDataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, list_IDs, type_data):
        self.type_data = type_data
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        #get labels
        if self.type_data == 'train':
            y = [xdf_dset.target.get(ID)]
            file = xdf_dset.destination_path.get(ID)
        elif self.type_data == 'test':
            y = [xdf_dset_test.target.get(ID)]
            file = xdf_dset_test.destination_path.get(ID)
        elif self.type_data == 'dev':
            y = [xdf_dset_dev.target.get(ID)]
            file = xdf_dset_dev.destination_path.get(ID)
        y= torch.FloatTensor(y)
        img = Image.open(file).convert('RGB')
        #img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        # X = torch.FloatTensor(img)
        preprocess = v2.Compose([
            v2.Resize(IMAGE_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        X = preprocess(img)
        X = torch.reshape(X, (CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        return X, y
class CustomDataLoader:
    def __init__(self):
        pass

    def read_data(self):
        list_of_ids = list(xdf_dset.index)
        list_of_ids_test = list(xdf_dset_test.index)
        list_of_ids_dev = list(xdf_dset_dev.index)

        partition = {
            'train': list_of_ids,
            'test': list_of_ids_test,
            'dev' : list_of_ids_dev
        }

        params = {'batch_size': BATCH_SIZE, 'shuffle': True}

        training_set = CustomDataset(partition['train'], 'train')
        training_generator = data.DataLoader(training_set, **params)

        params = {'batch_size': BATCH_SIZE, 'shuffle': False}
        test_set = CustomDataset(partition['test'], 'test')
        test_generator = data.DataLoader(test_set, **params)

        params = {'batch_size': BATCH_SIZE, 'shuffle': False}
        dev_set = CustomDataset(partition['dev'], 'dev')
        dev_generator = data.DataLoader(dev_set, **params)

        return training_generator, test_generator, dev_generator


