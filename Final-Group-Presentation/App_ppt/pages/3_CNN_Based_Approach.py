import streamlit as st

st.title('CNN Based Approach')

st.markdown('''
    - ResNet, a prominent CNN choice, initially employed but it has just 1 skip connection.
    - Tried densenet model as it offered multiple skip connections.
''')
st.image('densenet_vs_resnet.png', caption='Densenet Architecture VS Resnet Architecture')