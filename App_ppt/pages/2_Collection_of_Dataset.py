import streamlit as st

st.title('Collection of Datasets')

st.markdown('''
    - For fake images:
        - Collected datasets from 3 different sources:
            - GAN generated images ([link to the dataset](https://archive.org/details/1mFakeFaces))
            - GANPrintR generated images ([link to the research paper](https://www.researchgate.net/publication/337241864_GANprintR_Improved_Fakes_and_Evaluation_of_the_State_of_the_Art_in_Face_Manipulation_Detection))
            - Diffusion based models generated images ([link to the dataset](https://huggingface.co/datasets/OpenRL/DeepFakeFace))
    - For real images:
        - Collected datasets from 2 different sources:
            - CelebA-HQ-256 images ([link to the dataset](https://www.kaggle.com/datasets/denislukovnikov/celebahq256-images-only/) originally extracted from [PGGAN repo](https://github.com/tkarras/progressive_growing_of_gans))
            - Wiki images ([link to the dataset](https://huggingface.co/datasets/OpenRL/DeepFakeFace))
    - We had equal number of real and fake images.
    - A total of 120k images for our training, so no augmentation required.
    - Splitted the data into train, test, and dev.
        - Kept equal number of fake and real images.
        - For the fake images, we maintained an equal distribution across all datasets containing manipulated images.
''')

