import os
import torch
from PIL import Image
from torchvision.transforms import v2
from transformers import AutoImageProcessor
from torchvision.models import DenseNet
from torch import nn
import streamlit as st
import cv2
#%%
OR_PATH = os.getcwd()
sep = os.path.sep
os.chdir('..')
MODEL_DIR = os.getcwd() + sep + 'Models' + sep
os.chdir(OR_PATH)
#%%
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

torch.set_grad_enabled(False)
#%%
IMAGE_SIZE = 256
CHANNEL = 3
#%%


#%%
@st.cache_data
def preprocess_image(my_upload, model_type):
    image = Image.open(my_upload).convert('RGB')

    if model_type == 'CNN':
        processor = v2.Compose([
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        X = processor(image)
        X = torch.reshape(X, (CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
    if model_type == 'Transformer':
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        X = processor(images=image, return_tensors='pt')['pixel_values'].squeeze()

    return X

@st.cache_resource
def load_model(model_name):
    if model_name == "DenseNet":
        model = DenseNet(num_classes=1,
                         growth_rate=48,
                         num_init_features=64,
                         block_config=(6, 12, 24, 16)
                         )
        model.load_state_dict(torch.load(f'{MODEL_DIR}model_{model_name}.pt', map_location=device))
        model = model.to(device)
    return model

@st.cache_data
def model_inference(_X, _model):
    X = _X.unsqueeze(0).to(device)
    logits = _model(X)
    probability = nn.functional.sigmoid(logits).cpu().numpy()
    return probability

#%%
# TEST CASE
# test_image = 'test_img_01.png'
# image = Image.open(test_image)
#
# model_type = "CNN"
# model_name = "DenseNet"
#
# model = load_model(model_name)
# X = preprocess_image(image, model_type).to(device)
# probability = model_inference(X, model)
