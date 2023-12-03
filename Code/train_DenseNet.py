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
parser.add_argument('-c', action='store_true')
parser.add_argument('-e', '--excel', default='fully_processed.xlsx', type=str)
parser.add_argument('-n', '--name', default='DenseNet',type=str)
parser.add_argument('--dry', action='store_false')
args = parser.parse_args()

#%%
# EXCEL_FILE = 'fully_processed.xlsx' # --
EXCEL_FILE = args.excel
# CONTINUE_TRAINING = False
CONTINUE_TRAINING = args.c
# %%
IMAGE_SIZE = 256
CHANNEL = 3
BATCH_SIZE = 128 # --
# %%
# MODEL_NAME = 'DenseNet' # --
MODEL_NAME = args.name
# SAVE_MODEL = True # --
SAVE_MODEL = args.dry
N_EPOCHS = 20 # --
LR = 0.01 # --
MOMENTUM = 0.9 # --
ES_PATIENCE = 5 # --
LR_PATIENCE = 1 # --
SAVE_ON = 'AUROC' #--

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
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, verbose=True)

    print(model, file=open(f'summary_{MODEL_NAME}.txt', 'w'))

    return model, optimizer, criterion, scheduler


# %%

def train_test(train_gen, test_gen, metrics_lst, metric_names, save_on, early_stop_patience):

    save_on = metric_names.index(save_on)

    model, optimizer, criterion, scheduler = model_definition()
    sig = nn.Sigmoid()

    train_loss_item = list([])
    test_loss_item = list([])

    train_loss_hist = list([])
    test_loss_hist = list([])

    output_arr_hist = list([])
    pred_label_hist = list([])

    train_metrics_hist = list([])
    test_metrics_hist = list([])

    met_test_best = 0
    model_save_epoch = []

    if CONTINUE_TRAINING:
        model.load_state_dict(torch.load(f'model_{MODEL_NAME}.pt', map_location=device))
        model = model.to(device)
        print(f'Continuing Training - {MODEL_NAME}')
        model_save_epoch.append(0)
        model.eval()

        test_loss = 0
        steps_test = 0

        test_target_hist = list([])
        test_pred_labels = np.zeros(1)

        with tqdm(total=len(test_gen), desc=f'Saved Model') as pbar:
            with torch.no_grad():
                for xdata, xtarget in test_gen:
                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    optimizer.zero_grad()
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
                    pbar.set_postfix_str(f'Test  Loss: {avg_test_loss:.5f}')

            test_loss_hist.append(avg_test_loss)
            test_metrics = [metric.compute() for metric in metrics_lst]
            test_metrics_hist.append(test_metrics)
            _ = [metric.reset() for metric in metrics_lst]

            met_test = test_metrics[save_on]
            met_test_best = met_test

        xstrres = 'Saved Model:'
        for name, value in zip(metric_names, test_metrics):
            xstrres = f'{xstrres} Test {name} {value:.5f}'
        print(xstrres)

    for epoch in range(N_EPOCHS):
        train_loss = 0
        steps_train = 0

        train_target_hist = list([])


        # --Start Model Training--
        model.train()

        with tqdm(total=len(train_gen), desc=f'Epoch {epoch}') as pbar:
            for xdata, xtarget in train_gen:
                xdata, xtarget = xdata.to(device), xtarget.to(device)

                optimizer.zero_grad()
                output = model(xdata)
                loss = criterion(output, xtarget)
                loss.backward()
                optimizer.step()

                steps_train += 1
                train_loss += loss.item()
                train_loss_item.append([epoch, loss.item()])

                output_arr = nn.functional.sigmoid(output.detach().cpu()).numpy()

                if len(output_arr_hist) == 0:
                    output_arr_hist = output_arr
                else:
                    output_arr_hist = np.vstack([output_arr_hist, output_arr])

                if len(train_target_hist) == 0:
                    train_target_hist = xtarget.cpu().numpy()
                else:
                    train_target_hist = np.vstack([train_target_hist, xtarget.cpu().numpy()])

                pred_logit = output.detach().cpu()
                # pred_label = torch.round(pred_logit)
                pred_label = torch.where(pred_logit > 0.5, 1, 0)

                metrics_ = [metric(pred_label, xtarget.cpu()) for metric in metrics_lst]

                pbar.update(1)
                avg_train_loss = train_loss / steps_train
                pbar.set_postfix_str(f'Train Loss: {avg_train_loss:.5f}')

            train_loss_hist.append(avg_train_loss)
            train_metrics = [metric.compute() for metric in metrics_lst]
            train_metrics_hist.append([metric.compute() for metric in metrics_lst])
            _ = [metric.reset() for metric in metrics_lst]
        # --End Model Training--

        # --Start Model Test--
        test_loss = 0
        steps_test = 0

        test_target_hist = list([])

        test_pred_labels = np.zeros(1)

        model.eval()

        with tqdm(total=len(test_gen), desc=f'Epoch {epoch}') as pbar:
            with torch.no_grad():
                for xdata, xtarget in test_gen:
                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    optimizer.zero_grad()
                    output = model(xdata)
                    loss = criterion(output, xtarget)

                    steps_test += 1
                    test_loss += loss.item()
                    test_loss_item.append([epoch, loss.item()])

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
                    pbar.set_postfix_str(f'Test  Loss: {avg_test_loss:.5f}')

            test_loss_hist.append(avg_test_loss)
            test_metrics = [metric.compute() for metric in metrics_lst]
            test_metrics_hist.append(test_metrics)
            _ = [metric.reset() for metric in metrics_lst]

            met_test = test_metrics[save_on]

        xstrres = f'Epoch {epoch}'
        for name, value in zip(metric_names, train_metrics):
            xstrres = f'{xstrres} Train {name} {value:.5f}'

        xstrres = xstrres + ' - '
        for name, value in zip(metric_names, test_metrics):
            xstrres = f'{xstrres} Test {name} {value:.5f}'

        print(xstrres)

        # Save Best Model
        if met_test > met_test_best and SAVE_MODEL:
            torch.save(model.state_dict(), f'model_{MODEL_NAME}.pt')

            xdf_dset_results = xdf_dset_test.copy()  # global var
            xdf_dset_results['results'] = test_pred_labels[1:]

            xdf_dset_results.to_excel(f'results_{MODEL_NAME}.xlsx', index=False)
            print('Model Saved !!')
            met_test_best = met_test
            model_save_epoch.append(epoch)

        # Early Stopping
        if epoch - model_save_epoch[-1] > early_stop_patience:
            print('Early Stopping !! ')
            break


# %%
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

    early_stop_patience = ES_PATIENCE
    save_on = 'AUROC'
    train_test(train_gen, test_gen, metric_lst, metric_names, save_on, early_stop_patience)
