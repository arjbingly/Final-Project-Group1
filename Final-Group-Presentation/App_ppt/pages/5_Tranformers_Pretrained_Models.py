import streamlit as st


st.title('Transformers Based Models')

st.markdown('''
    - facebook/deit-base-distilled-patch16-384 ([link to the original paper](https://arxiv.org/abs/2012.12877))
        - Distilled data-efficient Image Transformer (DeiT) model pre-trained at resolution 224x224.
        - Fine-tuned at resolution 384x384 on ImageNet-1k (1 million images, 1,000 classes).
        
    - google/vit-base-patch16-224 ([link to the original paper](https://arxiv.org/abs/2010.11929))
        - Vision Transformer (ViT) model pre-trained on ImageNet-21k (14 million images, 21,843 classes) at resolution 224x224.
        - Fine-tuned on ImageNet 2012 (1 million images, 1,000 classes) at resolution 224x224.
        
    - google/vit-base-patch16-224 performed better than facebook/deit-base-distilled-patch16-384 and also training time for the facebook's distilled model was high, hence we discarded the model for our demo.
''')