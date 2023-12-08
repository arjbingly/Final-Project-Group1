import streamlit as st


st.title('Transformer Based Approach')

st.markdown('''
    - CNN becomes very model specific based on the data it is trained on.
    - To counter this problem, we use latent vector approach from an encoder trained on huge images dataset. ([link to the paper](https://arxiv.org/pdf/2302.10174.pdf))
    - We use the latent vector's output to further classify the images.
    - To practically approach this problem, we used two pretrained transformer based models.
''')