import streamlit as st
import pandas as pd

st.title('Results on Deit based models and CNN Based models')

st.markdown('Results of Deit based models:')
data_deit = {
    'Model': ['deit', 'deit_iFakeFaceDB'],
    'Accuracy':[0.83607, 0.66425],
    'Precision': [0.82924,0.61683],
    'Recall': [0.87737, 1.00000],
    'AUROC': [0.83243, 0.63467],
    'F1Score': [0.85262, 0.76301]
}
df_deit = pd.DataFrame(data_deit)
st.dataframe(df_deit)

st.markdown('Results of CNN based models:')
data_densenet = {
    'Model': ['Densenet', 'Densenet_diffusion', 'Densenet_GAN', 'Densenet_GANPrintR'],
    'Accuracy': [0.78239, 0.59407, 0.54048, 0.66416],
    'Precision': [0.86374, 0.57277, 0.54054, 0.61676],
    'Recall': [0.70928, 0.97970, 0.99883, 1.00000],
    'AUROC': [0.7883, 0.56010, 0.50010, 0.63457],
    'F1Score': [0.77892, 0.72291, 0.70146, .76296]
}
df_dense = pd.DataFrame(data_densenet)
st.dataframe(df_dense)

st.markdown('Results of ViT based models:')
data_vit = {
    'Model': ['ViT', 'ViT_diffusion', 'ViT_GAN', 'ViT_GANPrintR'],
    'Accuracy': [0.83616, 0.75791, 0.56116, 0.66407],
    'Precision': [0.84791, 0.75311, 0.55192, 0.61670],
    'Recall': [0.84919, 0.82134, 0.99950, 1.00000],
    'AUROC': [0.83501, 0.75232, 0.52254, 0.63447],
    'F1Score': [0.84885, 0.78575, 0.71115, .76291]
}
df_vit = pd.DataFrame(data_vit)
st.dataframe(df_vit)