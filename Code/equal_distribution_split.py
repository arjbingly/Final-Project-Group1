import pandas as pd
import random
from move_train_test_split import move_train_test_dev
data = pd.read_excel('image_data.xlsx',index_col=0)
OUTPUT_EXCEL = 'equal_distribution.xlsx'
# Filtering fake images based on the given folder distribution
fake_images = pd.DataFrame(columns=data.columns)
fake_folders = {'inpainting': 30000, 'insight': 30000, 'text2img': 30000, '1m_faces_00': 10000}
fake_count = 0

for folder, count in fake_folders.items():
    filtered_fake = data[(data['target_class'] == 'fake') & (data['folder'] == folder)].sample(n=count, random_state=42)
    fake_images = fake_images.append(filtered_fake)
    fake_count += count

# Filtering real images based on the given folder distribution
real_images = pd.DataFrame(columns=data.columns)
real_folders = {'celebahq256_imgs': 30000, 'wiki': 30000}
real_count = 0

for folder, count in real_folders.items():
    filtered_real = data[(data['target_class'] == 'real') & (data['folder'] == folder)].sample(n=count, random_state=42)
    real_images = real_images.append(filtered_real)
    real_count += count

# Balancing the number of fake and real images
if fake_count > real_count:
    fake_images = fake_images.sample(n=real_count, random_state=42)
else:
    real_images = real_images.sample(n=fake_count, random_state=42)

selected_data = pd.concat([fake_images, real_images])
selected_data = selected_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Creating splits (train - 70%, test - 20%, dev - 10%)
train_size = int(0.7 * len(selected_data))
test_size = int(0.2 * len(selected_data))
dev_size = len(selected_data) - train_size - test_size

selected_data.loc[:train_size, 'split'] = 'train'
selected_data.loc[train_size:(train_size + test_size), 'split'] = 'test'
selected_data.loc[(train_size + test_size):, 'split'] = 'dev'


selected_data.to_excel(OUTPUT_EXCEL, index=False)
move_train_test_dev(OUTPUT_EXCEL)
print(f'{selected_data.target_class.value_counts()}')
print(f'{selected_data.split.value_counts()}')
train_set = selected_data[selected_data['split']=='train']
test_set = selected_data[selected_data['split']=='test']
dev_set = selected_data[selected_data['split']=='dev']


real_train_set = train_set[train_set['target_class']=='real']
print(f'Inside train :\n{train_set.folder.value_counts()}')
print(f'Inside train :\n{train_set.target_class.value_counts()}')

print(f'Inside test :\n{test_set.folder.value_counts()}')
print(f'Inside test :\n{test_set.target_class.value_counts()}')

print(f'Inside dev :\n{dev_set.folder.value_counts()}')
print(f'Inside dev :\n{dev_set.target_class.value_counts()}')