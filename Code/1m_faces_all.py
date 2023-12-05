import os
import pandas as pd
import zipfile
# Function to get image paths recursively
OR_PATH = os.getcwd()
sep = os.path.sep
os.chdir('..')
DATA_DIR = os.getcwd() + sep + 'Data' + sep
EXCEL_DIR = os.getcwd() + sep + 'Excel' + sep
os.chdir(OR_PATH)
#%%
#EXCEL_FILE = EXCEL_DIR + args.excel
EXCEL_FILE = 'image_data_1m_face.xlsx'
def get_image_paths_in_folder(root_dir):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add or modify extensions as needed
                image_paths.append(os.path.join(root, file))
    return image_paths
def unzip_files(directory, files_to_unzip):
    for file in files_to_unzip:
        file_path = os.path.join(directory, file)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(directory)
def create_dataframe(root_dirs):
    data = {'image path': [], 'split': [], 'target_class': [], 'folder':[]}

    for root_dir in root_dirs:
            #faces_00_images = get_image_paths_in_folder(os.path.join(root_dir, '1m_faces_00'))
            data['image path'].extend(faces_00_images)
            data['split'].extend(['pending'] * len(faces_00_images))
            data['target_class'].extend(['fake'] * len(faces_00_images))
            data['folder'].extend(['1m_faces_00'] * len(faces_00_images))

    df = pd.DataFrame(data)
    return df

directories = [
    os.path.join('..', 'Data', 'downloaded_images', '1m_faces_00'),
    os.path.join('..', 'Data', 'downloaded_images', '1m_faces_01'),
    os.path.join('..', 'Data', 'downloaded_images', '1m_faces_02'),
    os.path.join('..', 'Data', 'downloaded_images', '1m_faces_03')
]
df = create_dataframe(directories)

df.to_excel('image_data_1m_face.xlsx', index=True)
print(df['target_class'].value_counts())
print(df['folder'].value_counts())
print(df.head())
