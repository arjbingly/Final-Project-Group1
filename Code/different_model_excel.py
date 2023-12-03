import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
import os
np.random.seed(42)
excel_folder_name = 'Excel'

data = pd.read_excel('image_data.xlsx',index_col=0)

fake_images = pd.DataFrame(columns=data.columns)
dict_total = {'inpainting': 20000, 'insight': 20000, 'text2img': 20000,'1m_faces_00': 10000,'iFakeFaceDB': 60000,'celebahq256_imgs': 30000, 'wiki': 30000}
diffusion_folders = ['inpainting', 'text2img', 'insight']
real_folders = ['celebahq256_imgs', 'wiki']

df_diffusion = pd.DataFrame()
df_real = pd.DataFrame()
df_1m_faces_00 = pd.DataFrame()
df_iFakeFaceDB = pd.DataFrame()

for folder, count in dict_total.items():
    if folder in diffusion_folders:
        folder_data = data[data['folder'] == folder]
        sampled_indx_diffusion = np.random.choice(folder_data.index, min(count, len(folder_data)), replace=False)
        sampled_data_diffusion = folder_data.loc[sampled_indx_diffusion]
        df_diffusion = pd.concat([df_diffusion, sampled_data_diffusion])
    elif folder in real_folders:
        folder_data = data[data['folder'] == folder]
        sampled_indx_real = np.random.choice(folder_data.index, min(count, len(folder_data)), replace=False)
        sampled_data_real = folder_data.loc[sampled_indx_real]
        df_real = pd.concat([df_real, sampled_data_real])
    elif folder == '1m_faces_00':
        folder_data = data[data['folder'] == folder]
        sampled_indx_1m_faces_00 = np.random.choice(folder_data.index, min(count, len(folder_data)), replace=False)
        sampled_data_1m_faces_00 = folder_data.loc[sampled_indx_1m_faces_00]
        df_1m_faces_00 = pd.concat([df_1m_faces_00, sampled_data_1m_faces_00])
    elif folder == 'iFakeFaceDB':
        folder_data_iFakeFaceDB = data[data['folder'] == folder]
        sampled_indx_iFakeFaceDB = np.random.choice(folder_data_iFakeFaceDB.index, min(count, len(folder_data_iFakeFaceDB)), replace=False)
        sampled_data_iFakeFaceDB = folder_data_iFakeFaceDB.loc[sampled_indx_iFakeFaceDB]
        df_iFakeFaceDB = pd.concat([df_iFakeFaceDB, sampled_data_iFakeFaceDB])

df_diffusion.reset_index(drop=True, inplace=True)
df_real.reset_index(drop=True, inplace=True)
df_1m_faces_00.reset_index(drop=True, inplace=True)
df_iFakeFaceDB.reset_index(drop=True, inplace=True)

df_total_diffusion = pd.concat([df_diffusion, df_real])
df_total_iFakeFaceDB = pd.concat([df_iFakeFaceDB, df_real])

small_real_df = df_real.sample(n=10000, random_state=42)
small_real_df.reset_index(drop=True, inplace=True)
df_total_1m_faces_00 = pd.concat([small_real_df, df_1m_faces_00])


def create_train_test_split(model_df,model_name = 'model_name'):
    shuffled_df = model_df.sample(frac=1, random_state=42)
    train_data, test_data = train_test_split(shuffled_df, test_size=0.3, random_state=42)
    train_data['split'] = 'train'
    test_data['split'] = 'test'
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    combined_data['target'] = combined_data['target_class'].apply(lambda x: 1 if x == 'real' else 0)
    combined_data['destination_path'] = combined_data['image path']
    combined_data = combined_data.rename(columns = {"image path": "image_path"})

    if not os.path.exists(excel_folder_name):
        os.makedirs(excel_folder_name)
    file_name = f'{model_name}.xlsx'
    file_path = os.path.join(excel_folder_name, file_name)
    combined_data.to_excel(file_path, index=False)
    return combined_data

data_diffusion =create_train_test_split(df_total_diffusion,'diffusion')
data_iFakeFaceDB =create_train_test_split(df_total_iFakeFaceDB,'iFakeFaceDB')
data_1m_faces_00 =create_train_test_split(df_total_1m_faces_00,'1m_faces_00')

