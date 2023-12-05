import os
import tarfile
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
EXCEL_FILE = '../Excel/image_data_1m_face.xlsx'
output_excel = '../Excel/1m_faces_with_real.xlsx'
def get_image_paths_in_folder(root_dir):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add or modify extensions as needed
                image_path = os.path.join(root, file)
                # Check if the image file is valid
                try:
                    _ = Image.open(image_path).convert('RGB')
                    image_paths.append(image_path)
                except (IOError, SyntaxError):
                    print(f"Corrupted image: {image_path}")
    return image_paths
def unzip_files(directory, files_to_unzip):
    for file in files_to_unzip:
        file_path = os.path.join(directory, file)
        if file.lower().endswith('.tar'):
            with tarfile.open(file_path, 'r') as tar_ref:
                tar_ref.extractall(directory)
        os.remove(file_path)

def create_dataframe(root_dirs):
    data = {'image_path': [],'destination_path' :[] ,'folder' :[] ,'split': [], 'target_class': [], 'target': []}
    for root_dir in root_dirs:
        for root, dirs, files in os.walk(root_dir):
            #if dirs.isin(['1m_faces_01']):
            for folder in dirs:
                images_in_folder = get_image_paths_in_folder(os.path.join(root, folder))
                num_images = len(images_in_folder)
                data['image_path'].extend(images_in_folder)
                data['destination_path'].extend(images_in_folder)
                data['folder'].extend(['1mFakeFaces'] * num_images)
                data['split'].extend(['pending'] * num_images)
                data['target_class'].extend(['fake'] * num_images)
                data['target'].extend([0] * num_images)

                print(len(data['image_path']))
                print(len(data['destination_path']))
                print(len(data['folder']))
                print(len(data['split']))
                print(len(data['target_class']))
                print(len(data['target']))
    df = pd.DataFrame(data)
    return df

directories = [
    os.path.join('..','Data', 'downloaded_images', '1mFakeFaces'),

]
tar_files = [file for file in os.listdir(directories[0]) if file.lower().endswith('.tar')]
unzip_files(directories[0], tar_files)
#df = create_dataframe(directories)
#df.to_excel(EXCEL_FILE, index=False)
# df = pd.read_excel(EXCEL_FILE)
# df.reset_index(drop=True, inplace=True)
# df_60k = df.head(60000)
# data = pd.read_excel('../Excel/fully_processed.xlsx')
# celeb = data[data['folder'] == 'celebahq256_imgs']
# wiki = data[data['folder'] == 'wiki']
# combined_df = pd.concat([celeb, wiki], ignore_index=True)
#
# full_df = pd.concat([df_60k, combined_df], ignore_index=True)
# shuffled_df = full_df.sample(frac=1, random_state=42)  # frac=1 shuffles all rows


train_df, test_df = train_test_split(shuffled_df, test_size=0.3, random_state=42)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
train_df['split'] = 'train'
test_df['split'] = 'test'
final_df = pd.concat([train_df, test_df])
final_df.to_excel(output_excel, index=False)
# df = pd.
