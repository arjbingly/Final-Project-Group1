import argparse
import numpy as np
import pandas as pd
import torch
import os
from torchvision.models import DenseNet
from torch.utils import data
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from tqdm import tqdm
from load_data import CustomDataset, CustomDataLoader
# %%
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--excel', default='fully_processed.xlsx', type=str)
parser.add_argument('-n', '--name', default='DenseNet',type=str)
parser.add_argument('-s', '--split', choices=['train', 'test', 'dev'], default='test')

args = parser.parse_args()

#%%
# EXCEL_FILE = 'fully_processed.xlsx' # --
EXCEL_FILE = args.excel
# %%
IMAGE_SIZE = 256
CHANNEL = 3
BATCH_SIZE = 64
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

# %%
OR_PATH = os.getcwd()
sep = os.path.sep
os.chdir('..')
DATA_DIR = os.getcwd() + sep + 'Data' + sep
os.chdir(OR_PATH)


# %%
def model_definition():
    model = DenseNet(num_classes=1,
                     growth_rate=48,
                     num_init_features=64,
                     block_config=(6, 12, 24, 16)
                     )
    model.load_state_dict(torch.load(f'model_{MODEL_NAME}.pt', map_location=device))
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

                output = model(xdata)
                loss = criterion(output, xtarget)

                steps_test += 1
                test_loss += loss.item()

                output_arr = output.detach().cpu().numpy()

                if len(test_target_hist) == 0:
                    test_target_hist = xtarget.cpu().numpy()
                else:
                    test_target_hist = np.vstack([test_target_hist, xtarget.cpu().numpy()])

                pred_logit = output.detach().cpu()
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
