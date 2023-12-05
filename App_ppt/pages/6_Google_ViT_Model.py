import streamlit as st

st.title('Google Vision Transfomer Model')

st.markdown('''
    - Images are presented to the model as a sequence of fixed-size patches (resolution 16x16) which are linearly embedded.
    - By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks.    
''')

st.image('ViTClassifierArchitecture.png', caption="Google's ViT Architecture")