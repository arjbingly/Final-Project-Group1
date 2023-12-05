import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ast

model_ = 'DenseNet'
vit_diffusion_excel = f'tsne_cluster_{model_}_diffusion.xlsx'
vit_GAN_PrintR_excel = f'tsne_cluster_{model_}_GAN_PrintR.xlsx'
vit_GAN_excel = f'tsne_cluster_{model_}_GAN.xlsx'
vit_broad_excel = f'tsne_cluster_{model_}_broad.xlsx'


#input_excel_file = vit_GAN_PrintR_excel

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
excel_folder = os.path.join(parent_dir, 'Excel')
def string_to_list(string):
    values = re.findall(r"[-+]?\d*\.\d+|\d+", string)
    return [float(val) if '.' in val else int(val) for val in values]

def create_plot(title):
    #data['embedding'] = data['embedding'].apply(ast.literal_eval)
    data['embedding'] = data['embedding'].apply(string_to_list)
    max_length = max(map(len, data['embedding']))
    padded_embeddings = pad_sequences(data['embedding'], maxlen=max_length, padding='post', truncating='post',
                                      dtype='float32')
    # Convert to NumPy array
    embeddings_array = np.array(padded_embeddings)
    #embeddings = np.array(data['embedding'].tolist())
    embeddings = embeddings_array
    labels = data['dataset'].tolist()
    tsne = TSNE(n_components=2, random_state=42)
    transformed_features = tsne.fit_transform(embeddings)
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

# Plotting
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        indices = np.where(np.array(labels) == label)
        plt.scatter(
            transformed_features[indices, 0],
            transformed_features[indices, 1],
            color=colors(i),
            label=label
        )
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    file_name = title + '.png'
    #, dpi=300
    plt.savefig(file_name)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='model itself ')
    # parser.add_argument('-m','--name', type=str, help='broad gan diffusion gan ganprintr')
    # args = parser.parse_args()
    # name = args.name
    #name = 'broad'
    if name  == 'diffusion':
        input_excel_file = vit_broad_excel
        title = f'{model_} embeddings on broad'
    elif name == 'diffusion':
        input_excel_file = vit_diffusion_excel
        title =f'{model_} embeddings on diffusion'
    elif name == 'gan':
        input_excel_file = vit_GAN_excel
        title = f'{model_} embeddings on GAN '
    elif name == 'ganprintr':
        input_excel_file = vit_GAN_PrintR_excel
        title = f'{model_} embeddings on GAN PrintR'
    input_excel_path = os.path.join(excel_folder, input_excel_file)
    data = pd.read_excel(input_excel_path)
    create_plot(title)




