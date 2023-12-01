import pandas as pd
import shutil
import os

def move_train_test_dev(excel_path = 'equal_distribution.xlsx'):
    data = pd.read_excel(excel_path)
    DATA_FOLDER = 'Data'
    for index, row in data.iterrows():
        image_path = row['image path']
        split = row['split']
        destination_folder = os.path.join('..', DATA_FOLDER, split)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        try:
            image_folder = os.path.dirname(image_path)
            shutil.copy(image_path, destination_folder)
            print(f"Copied images from {image_folder} to {destination_folder}")
        except FileNotFoundError:
            print(f"File {image_path} not found.")
        except Exception as e:
            print(f"Error occurred while copying images: {str(e)}")
move_train_test_dev()