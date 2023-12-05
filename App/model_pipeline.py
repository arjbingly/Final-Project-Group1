import os
import torch
from PIL import Image
from torchvision.transforms import v2
from transformers import AutoImageProcessor, AutoModelForImageClassification
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
class DenseNet_pipeline():
    def __init__(self, model_name):
        self.model_name = model_name
        self.load_model()

    def preprocess_image(self, my_upload):
        image = Image.open(my_upload).convert('RGB')
        processor = v2.Compose([
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        X = processor(image)
        X = torch.reshape(X, (CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        return X
    def load_model(self):
        self.model = DenseNet(num_classes=1,
                         growth_rate=48,
                         num_init_features=64,
                         block_config=(6, 12, 24, 16)
                         )
        self.model.load_state_dict(torch.load(f'{MODEL_DIR}model_{self.model_name}.pt', map_location=device))
        self.model = self.model.to(device)

    def inference(self,my_upload):
        X = self.preprocess_image(my_upload)
        X = X.unsqueeze(0).to(device)
        logits = self.model(X)
        probability = nn.functional.sigmoid(logits).cpu().numpy()
        return probability[0,0]

#%%
class VIT_pipeline():
    def __init__(self, model_name):
        self.model_name = model_name
        self.load_model()

    def preprocess_image(self, my_upload):
        image = Image.open(my_upload).convert('RGB')
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        X = processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
        return X
    def load_model(self):
        self.model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224",
                                                                num_labels=1,
                                                                ignore_mismatched_sizes=True)
        self.model.load_state_dict(torch.load(f'{MODEL_DIR}model_{self.model_name}.pt', map_location=device))
        self.model = self.model.to(device)

    def inference(self,my_upload):
            X = self.preprocess_image(my_upload)
            X = X.unsqueeze(0).to(device)
            logits = self.model(X).logits
            probability = nn.functional.sigmoid(logits).cpu().numpy()
            return probability[0,0]
#%%

#%%
# # TEST CASE
# test_image = 'test_img_01.png'
#
# model_name = "VIT_diffusion"
#
# model_pipe = VIT_pipeline(model_name)
# probability= model_pipe.inference(test_image)
# print(probability)
