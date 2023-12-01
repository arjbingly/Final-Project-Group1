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
            return True
    except FileNotFoundError:
        print(f"File {image_path} not found.")
    except Exception as e:
        print(f"Error occurred while resizing and copying images: {str(e)}")


def move_train_test_dev(excel_path = 'equal_distribution.xlsx'):
    data = pd.read_excel(excel_path)
    DATA_FOLDER = 'Data'
    output_excel = 'final_data.xlsx'

    #final_data = pd.DataFrame(columns=['image_path', 'destination_folder', 'folder', 'target_class'])
    image_paths = []
    new_paths = []
    folders = []
    splits = []
    target_classes = []

    for index, row in data.iterrows():
        image_path = row['image path']
        split = row['split']
        target_class = row['target_class']
        folder = row['folder']
        destination_folder = os.path.join('..', DATA_FOLDER, split)
        new_image_path = os.path.join(destination_folder, os.path.basename(image_path))
        #destination_path = os.path.join('..', destination_folder, split)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        try:
            image_folder = os.path.dirname(image_path)
            #shutil.copy(image_path, destination_folder)
            success = resize_and_copy_image(image_path, destination_folder)
            if success:
                image_paths.append(image_path)
                new_paths.append(new_image_path)
                folders.append(folder)
                splits.append(split)
                target_classes.append(target_class)
            print(f"Copied images from {image_folder} to {destination_folder}")
        except FileNotFoundError:
            print(f"File {image_path} not found.")
        except Exception as e:
            print(f"Error occurred while copying images: {str(e)}")
    final_data = pd.DataFrame({
        'image_path': image_paths,
        'destination_path': new_paths,
        'folder': folders,
        'split': splits,
        'target_class': target_classes
    })

    final_data.to_excel(output_excel, index=False)
    print(f"Generated {output_excel} file.")
#move_train_test_dev()

def process_target():
    output_excel = 'fully_processed.xlsx'
    input_excel = 'final_data.xlsx'
    data = pd.read_excel(input_excel)
    image_paths = []
    new_paths = []
    folders = []
    splits = []
    target_classes = []
    targets = []
    for index, row in data.iterrows():
        image_paths.append(row['image_path'])
        new_paths.append(row['destination_path'])
        splits.append(row['split'])
        target_classes.append(row['target_class'])
        folders.append(row['folder'])
        if row['target_class'] == 'real':
            targets.append(1)
        else:
            targets.append(0)

    final_data = pd.DataFrame({
        'image_path': image_paths,
        'destination_path': new_paths,
        'folder': folders,
        'split': splits,
        'target_class': target_classes,
        'target' : targets
    })
    final_data.to_excel(output_excel, index=False)
process_target()