import numpy as np
import torch
import os
from torchvision.models import DenseNet
from torch.utils import data
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
import tqdm

#%%
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
#%%
MODEL_NAME = 'DenseNet'
SAVE_MODEL = True
LR = 0.0001
MOMENTUM = 0.9
def model_definition():
    model = DenseNet(num_classes=1,
                     growth_rate=48,
                     num_init_features=64,
                     block_config=(6, 12, 24, 16)
                     )
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    print(model, file=open(f'summary_{MODEL_NAME}.txt', 'w'))

    return model, optimizer, criterion, scheduler

#%%
# test torchmetreics
# metrics accuracy, auroc, precision, recall, f1_score,
metrics = [torchmetrics.Accuracy(task='binary'),
           torchmetrics.Precision(task='binary'),
           torchmetrics.Recall(task='binary'),
           torchmetrics.AUROC(task='binary'),
           torchmetrics.F1Score(task='binary')]

for epoch in range(2):
    preds_list = [[0,1,0],[1,1,1],[0,0,0]]
    target_list = [[0,1,0],[0,1,0],[0,1,0]]
    for i in range(3):
        preds = torch.Tensor(preds_list[i])
        target = torch.Tensor(target_list[i])
        metrics_ = [metric(preds,target) for metric in metrics]
        print(f'{i}: accs={metrics_}')

    metrics_ = [metric.compute() for metric in metrics]
    _ = [metric.reset() for metric in metrics]
    print(f'Overall Acc : {metrics_}')

#%%
N_EPOCHS = 10
def train_test(train_gen, test_gen, metrics_lst, metric_names, save_on, early_stop_patience):

    save_on = metric_names.index(save_on)

    model, optimizer, criterion, scheduler = model_definition()

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

    for epoch in range(N_EPOCHS):
        train_loss = 0
        steps_train = 0

        train_target_hist = list([])

        #--Start Model Training--
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

                output_arr = output.detach().cpu().numpy()

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
                pred_label = torch.where(pred_logit > 0.5, 1 ,0)

                metrics_ = [metric(pred_label, xtarget.cpu()) for metric in metrics_lst]

                pbar.update(1)
                avg_train_loss = train_loss/steps_train
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
                    output.model(xdata)
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

                    test_pred_labels = np.vstack(test_pred_labels, pred_label.numpy())

                    metrics_ = [metric(pred_label, xtarget.cpu()) for metric in metrics_lst]

                    pbar.update(1)
                    avg_test_loss = test_loss / steps_test
                    pbar.set_postfix_str(f'Test  Loss: {avg_train_loss:.5f}')

            test_loss_hist.append(avg_test_loss)
            test_metrics = [metric.compute() for metric in metrics_lst]
            test_metrics_hist.append(test_metrics)
            _ = [metric.reset() for metric in metrics_lst]

            met_test = test_metrics[save_on]

        xstrres = f'Epoch {epoch}'
        for name,value in zip(metric_names, train_metrics):
            xstrres = f'{xstrres} Train {name} {value:.5f}'

        xstrres = xstrres + ' - '
        for name,value in zip(metric_names, test_metrics):
            xstrres = f'{xstrres} Test {name} {value:.5f}'

        print(xstrres)

        # Save Best Model
        if met_test > met_test_best and SAVE_MODEL:
            torch.save(model.state_dict(), f'model_{MODEL_NAME}.pt')

            xdf_dset_results = xdf_dset_test.copy() #global var
            xdf_dset_results['results'] = test_pred_labels

            xdf_dset_results.to_excel(f'results_{MODEL_NAME}', index = False)
            print('Model Saved !!')
            met_test_best = met_test
            model_save_epoch.append(epoch)

        # Early Stopping
        if epoch - model_save_epoch[-1] > early_stop_patience:
            print('Early Stopping !! ')
            break






