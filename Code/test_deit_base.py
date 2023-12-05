import argparse
import numpy as np
import pandas as pd
import torch
import os
from torchvision.models import DenseNet
from torch.utils import data
from torch import nn
import torchmetrics
from tqdm import tqdm
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoImageProcessor, AutoModelForImageClassification

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--excel', default='fully_processed.xlsx', type=str)
parser.add_argument('-n', '--name', default='DenseNet',type=str)
parser.add_argument('-s', '--split', choices=['train', 'test', 'dev'], default='test')
args = parser.parse_args()

# %%
OR_PATH = os.getcwd()
sep = os.path.sep
os.chdir('..')
DATA_DIR = os.getcwd() + sep + 'Data' + sep
EXCEL_DIR = os.getcwd() + sep + 'Excel' + sep
os.chdir(OR_PATH)
#%%
EXCEL_FILE = EXCEL_DIR + args.excel
# CONTINUE_TRAINING = False
# CONTINUE_TRAINING = args.c
# %%
IMAGE_SIZE = 256
CHANNEL = 3
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.9
PRETRAINED_MODEL ="facebook/deit-base-distilled-patch16-384"
# %%
# MODEL_NAME = 'DenseNet' # --
MODEL_NAME = args.name
SPLIT = args.split
# %%
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
#%%
#%%
# Load Data
xdf_data = pd.read_excel(EXCEL_FILE)
xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
xdf_dset_test = xdf_data[xdf_data["split"] == 'test'].copy()
xdf_dset_dev = xdf_data[xdf_data["split"] == 'dev'].copy()
class CustomDataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, list_IDs, type_data):
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL)


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
        X = self.processor(images=img, return_tensors='pt')['pixel_values'].squeeze()
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
# %%
def model_definition():
    model = AutoModelForImageClassification.from_pretrained(PRETRAINED_MODEL,
                                                            num_labels=1,
                                                            ignore_mismatched_sizes=True)
    # model.load_state_dict(torch.load(f'model_deit.pt', map_location=device))
    model.load_state_dict(torch.load(f'model_deit_iFakeFaceDB.pt', map_location=device))
    model = model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.BCEWithLogitsLoss()

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, verbose=True)

    print(model, file=open(f'summary_{MODEL_NAME}.txt', 'w'))

    return model, criterion


# %%
def eval_model(test_gen, metrics_lst, metric_names):
    model, criterion = model_definition()

    test_loss_hist = list([])
    test_metrics_hist = list([])

    # --Start Model Test--
    test_loss = 0
    steps_test = 0

    test_target_hist = list([])

    test_pred_labels = np.zeros(1)

    model.eval()
    print(f'-- Evaluating {MODEL_NAME} on {SPLIT} set --')
    with tqdm(total=len(test_gen), desc=f'Evaluating - ') as pbar:
        with torch.no_grad():
            for xdata, xtarget in test_gen:
                xdata, xtarget = xdata.to(device), xtarget.to(device)

                output = model(xdata).logits
                loss = criterion(output, xtarget)

                steps_test += 1
                test_loss += loss.item()

                output_arr = output.detach().cpu().numpy()

                if len(test_target_hist) == 0:
                    test_target_hist = xtarget.cpu().numpy()
                else:
                    test_target_hist = np.vstack([test_target_hist, xtarget.cpu().numpy()])

                pred_logit = output.detach().cpu()
                pred_logit = nn.functional.sigmoid(pred_logit)
                # pred_label = torch.round(pred_logit)
                pred_label = torch.where(pred_logit > 0.5, 1, 0)

                test_pred_labels = np.vstack([test_pred_labels, pred_label.numpy()])

                metrics_ = [metric(pred_label, xtarget.cpu()) for metric in metrics_lst]

                pbar.update(1)
                avg_test_loss = test_loss / steps_test
                pbar.set_postfix_str(f'Loss: {avg_test_loss:.5f}')

        test_loss_hist.append(avg_test_loss)
        test_metrics = [metric.compute() for metric in metrics_lst]
        test_metrics_hist.append(test_metrics)
        _ = [metric.reset() for metric in metrics_lst]

    xstrres = ''
    for name, value in zip(metric_names, test_metrics):
        xstrres = f'{xstrres} {name} {value:.5f}'
    print(xstrres)

    return test_pred_labels[1:], test_metrics
# %%
def save_results(pred_labels, test_metrics, metric_names):
    xdf_dset_results = xdf_data[xdf_data["split"] == SPLIT].copy()
    xdf_dset_results['results'] = pred_labels
    xdf_dset_results.to_excel(f'results_{MODEL_NAME}.xlsx', index=False)

    lines = ['*** Evaluation Results ***', f'Model Name : {MODEL_NAME}',f'Excel File : {EXCEL_FILE}', f'Split : {SPLIT}']
    xstrres = ''
    for name, value in zip(metric_names, test_metrics):
        xstrres = f'{xstrres} {name} {value:.5f}'
    lines.append(xstrres)

    with open(f'results_{MODEL_NAME}.txt', 'w') as f:
        f.writelines([f'{line} \n' for line in lines])

    print('Results Saved !!')
#%%
if __name__ == '__main__':
    xdf_data = pd.read_excel(EXCEL_FILE)
    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
    xdf_dset_test = xdf_data[xdf_data["split"] == 'test'].copy()
    xdf_dset_dev = xdf_data[xdf_data["split"] == 'dev'].copy()

    data_loader = CustomDataLoader()
    train_gen, test_gen, dev_gen = data_loader.read_data()

    metric_lst = [torchmetrics.Accuracy(task='binary'),
                  torchmetrics.Precision(task='binary'),
                  torchmetrics.Recall(task='binary'),
                  torchmetrics.AUROC(task='binary'),
                  torchmetrics.F1Score(task='binary')]
    metric_names = ['Accuracy',
                    'Precision',
                    'Recall',
                    'AUROC',
                    'F1Score']

    if SPLIT=='train':
        pred_labels, test_metrics = eval_model(train_gen, metric_lst, metric_names)
    elif SPLIT=='test':
        pred_labels, test_metrics = eval_model(test_gen, metric_lst, metric_names)
    elif SPLIT=='dev':
        pred_labels, test_metrics = eval_model(dev_gen, metric_lst, metric_names)

    save_results(pred_labels, test_metrics, metric_names)
