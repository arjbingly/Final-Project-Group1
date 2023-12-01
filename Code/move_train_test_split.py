import pandas as pd
import shutil
import os
from PIL import Image

def resize_and_copy_image(image_path, destination_folder, target_size=(256, 256)):
    try:
        # Open the image using PIL
        with Image.open(image_path) as img:
            # Resize the image
            resized_img = img.resize(target_size)
            # Save the resized image to the destination folder
            resized_img.save(os.path.join(destination_folder, os.path.basename(image_path)))
            print(f"Resized and copied images from {image_path} to {destination_folder}")
    except FileNotFoundError:
        print(f"File {image_path} not found.")
    except Exception as e:
        print(f"Error occurred while resizing and copying images: {str(e)}")

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
            #shutil.copy(image_path, destination_folder)
            resize_and_copy_image(image_path, destination_folder)
            print(f"Copied images from {image_folder} to {destination_folder}")
        except FileNotFoundError:
            print(f"File {image_path} not found.")
        except Exception as e:
            print(f"Error occurred while copying images: {str(e)}")
move_train_test_dev()