import os
import pandas as pd
import argparse
import torch
from PIL import Image
from transformers import AutoModelForImageClassification
from transformers import AutoImageProcessor
from torch import nn
#from torchvision.models import DenseNet
from transformers import AutoModelForImageClassification
from torchvision.models import DenseNet
from torchvision.transforms import v2
import torch.nn.functional as F
#PRETRAINED_MODEL ="google/vit-base-patch16-224"
IMAGE_SIZE = 256
CHANNEL = 3
CUR_MODEL = None #AutoModelForImageClassification.from_pretrained(PRETRAINED_MODEL, num_labels=1,
             #                                          ignore_mismatched_sizes=True)
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

torch.set_grad_enabled(False)

model_ = 'DenseNet'
model_name_diffusion = f'model_{model_}_diffusion.pt'
model_name_GAN_PrintR= f'model_{model_}_GAN_PrintR.pt'
model_name_broad_model = f'model_{model_}.pt'
model_name_GAN_model = f'model_{model_}_GAN.pt'

vit_diffusion_excel = f'tsne_cluster_{model_}_diffusion.xlsx'
vit_GAN_PrintR_excel = f'tsne_cluster_{model_}_GAN_PrintR.xlsx'
vit_GAN_excel = f'tsne_cluster_{model_}_GAN.xlsx'
vit_broad_excel = f'tsne_cluster_{model_}_broad.xlsx'

# model_name = model_name_diffusion
# new_excel_file = vit_diffusion_excel


current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
code_folder = os.path.join(parent_dir, 'Code')
excel_folder = os.path.join(parent_dir, 'Excel')
CUR_MODEL = None

excel_file = 'tsne_cluster.xlsx'
input_excel_path = os.path.join(excel_folder, excel_file)

# def read_results_file(name):
#     current_dir = os.getcwd()
#     parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
#     excel_folder = os.path.join(parent_dir, 'Excel')
#     results_file = os.path.join(excel_folder, f'results_{name}.xlsx')
#     if os.path.exists(results_file):
#         df = pd.read_excel(results_file)
#         return df
#     else:
#         print(f"File results_{name}.xlsx does not exist in the 'Excel' folder.")
#         return None
# def preprocess_image():
#     image = Image.open('test_img_01.png').convert('RGB')
#     #image = Image.open(image).convert('RGB')
#     processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL)
#     X = processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
#     return X
def get_embeddings(image_path):
    image = Image.open(image_path).convert('RGB')
    #processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL)
    #X = processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
    processor = v2.Compose([
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    X = processor(image)
    X = torch.reshape(X, (CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
    X = X.unsqueeze(0).to(device)

    prev_layer = CUR_MODEL(X)

    #embedding = prev_layer.cpu().numpy()
    global_avg_pool = nn.AdaptiveAvgPool2d(1)
    embeddings_pooled = global_avg_pool(prev_layer).squeeze()
    embeddings_flat = embeddings_pooled.view(embeddings_pooled.size(0), -1)
    embeddings_list = embeddings_flat.squeeze().tolist()

    return embeddings_list
def load_pytorch_model(model_path):


    # model = AutoModelForImageClassification.from_pretrained(PRETRAINED_MODEL, num_labels=1,
    #                                                    ignore_mismatched_sizes=True)

    print(model_path)

    if os.path.exists(model_path):
        model = DenseNet(num_classes=1,
                         growth_rate=48,
                         num_init_features=64,
                         block_config=(6, 12, 24, 16)
                         )
        #model.load_state_dict(torch.load(f'{MODEL_DIR}model_{model_name}.pt', map_location=device))

        model.load_state_dict(torch.load(model_path))
        model = nn.Sequential(*list(model.children())[:-1])
        model.cuda()
        return model
    else:
        print(f"PyTorch model {model_path} does not exist.")
        return None
# Usage with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model itself ')
    parser.add_argument('-m','--name', type=str, help='broad gan diffusion gan ganprintr')
    args = parser.parse_args()
    name = args.name
    #name = 'diffusion'

    if name  == 'broad':
        model_name = model_name_broad_model
        new_excel_file = vit_broad_excel
    elif name == 'diffusion':
        model_name = model_name_diffusion
        new_excel_file = vit_diffusion_excel
    elif name == 'gan':
        model_name = model_name_GAN_model
        new_excel_file = vit_GAN_excel
    elif name == 'ganprintr':
        model_name = model_name_GAN_PrintR
        new_excel_file = vit_GAN_PrintR_excel
    # name = 'deit'
    # db_name = 'iFakeFaceDB'

    model_path = os.path.join(code_folder, model_name)
    output_excel_path = os.path.join(excel_folder, new_excel_file)
    df = pd.read_excel(input_excel_path)

    #data = read_results_file(name)
    if df is not None:
        print("Data from Excel file:")
        print(df.head(5))
        CUR_MODEL = load_pytorch_model(model_path)
        if CUR_MODEL is not None:
            print(f'model loaded successfully')
        else:
            print(f'model loading failed')

        embeddings_list = []
        for index, row in df.iterrows():
            image_path = row['image_path']

            if image_path.split('/')[-2] == '1m_faces_00':
                image_path_list =image_path.split('/')
                image_path_list.remove('1m_faces_00')
                image_path = '/'.join(image_path_list)

            embedding = get_embeddings(image_path)
            #embed = embedding
            string_representation = '[' + ','.join(map(str, embedding)) + ']'
            embeddings_list.append(string_representation)
        df['embedding'] = embeddings_list

        df.to_excel(output_excel_path, index=False)
        print(f'excel file generated : {output_excel_path}')



