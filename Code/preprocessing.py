import os
import pandas as pd
import zipfile
# Function to get image paths recursively
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
        if root_dir.endswith('celebahq256_imgs'):
            image_paths = get_image_paths_in_folder(root_dir)
            data['image path'].extend(image_paths)
            data['split'].extend(['pending'] * len(image_paths))
            data['target_class'].extend(['real'] * len(image_paths))
            data['folder'].extend(['celebahq256_imgs'] * len(image_paths))

        elif root_dir.endswith('DeepFakeFace'):
            # Real images from wiki folder
            files_to_unzip = ['wiki.zip', 'inpainting.zip', 'insight.zip', 'text2img.zip']
            unzip_files(root_dir, files_to_unzip)


            wiki_images = get_image_paths_in_folder(os.path.join(root_dir, 'wiki'))
            data['image path'].extend(wiki_images)
            data['split'].extend(['pending'] * len(wiki_images))
            data['target_class'].extend(['real'] * len(wiki_images))
            data['folder'].extend(['wiki'] * len(image_paths))

            # Fake images from subfolders
            fake_dirs = ['inpainting', 'insight', 'text2img']
            for fake_dir in fake_dirs:
                fake_images = get_image_paths_in_folder(os.path.join(root_dir, fake_dir))
                data['image path'].extend(fake_images)
                data['split'].extend(['pending'] * len(fake_images))
                data['target_class'].extend(['fake'] * len(fake_images))
                data['folder'].extend([fake_dir] * len(fake_images))

        elif root_dir.endswith('1m_faces_00'):
            faces_00_images = get_image_paths_in_folder(os.path.join(root_dir, '1m_faces_00'))
            data['image path'].extend(faces_00_images)
            data['split'].extend(['pending'] * len(faces_00_images))
            data['target_class'].extend(['fake'] * len(faces_00_images))
            data['folder'].extend(['1m_faces_00'] * len(faces_00_images))

    df = pd.DataFrame(data)
    return df


directories = [
    os.path.join('..', 'Data', 'downloaded_images', 'celebahq256_imgs'),
    os.path.join('..', 'Data', 'downloaded_images', 'DeepFakeFace'),
    os.path.join('..', 'Data', 'downloaded_images', '1m_faces_00')
]

df = create_dataframe(directories)

df.to_excel('image_data.xlsx', index=True)
print(df['target_class'].value_counts())
print(df['folder'].value_counts())
print(df.head())
