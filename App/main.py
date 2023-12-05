import streamlit as st
from PIL import Image
import model_pipeline as mp
import torch
import plotly.express as px
import pandas as pd
import numpy as np
#%%
import matplotlib.pyplot as plt


def sidebar():
    with st.sidebar:
        model_kind = st.radio(
            "Model Kind : ",
            ["Broad Classifier", "Specific Classifier"],
            captions=["Classifies fake and real human faces from varity of models.",
                      "Classifies fake and real human faces and specifies the most likely model used to generate the image if fake.",],
            index=0,
        )
        model_type = st.selectbox(
            "Model Type :",
            ['CNN', 'Transformer'],
            # captions = ["Uses a Convolutional Neural Network based network to classify",
            #             "Uses a Transformer based model to classifer"],
            index = 0
        )

        if model_type == 'CNN':
            model_name = 'DenseNet'
        elif model_type == 'Transformer':
            model_name = 'VIT'

    return model_kind, model_name



def main():
    st.image(f'logo/black_logo.png')
    # st.header("Face Auth")
    st.divider()
    st.subheader('Image File Uploader')

    model_kind, model_name = sidebar()

    my_upload = st.file_uploader('Upload an image', type=['png','jpg','jpeg'])

    if model_kind == 'Broad Classifier':
        if model_name == 'DenseNet':
            model = mp.DenseNet_pipeline(model_name)
        if model_name == 'VIT':
            model = mp.VIT_pipeline(model_name)
    elif model_kind == 'Specific Classifier':
        model_subtypes = ['diffusion', 'GAN', 'GANprintR']
        model_names = [f'{model_name}_{subtype}' for subtype in model_subtypes]
        if model_name == 'DenseNet':
            models = [mp.DenseNet_pipeline(name) for name in model_names]
        if model_name == 'VIT':
            models = [mp.VIT_pipeline(name) for name in model_names]



    if my_upload is None:
        st.stop()

    st.image(Image.open(my_upload), caption='Uploaded Image', use_column_width=True)

    st.write(f"Model in use : **{model_name}**")
    clicked = st.button('Is that a Fake Face?')

    if clicked:
        torch.set_grad_enabled(False)

        if model_kind == 'Broad Classifier':
            probability = model.inference(my_upload)

            col1, col2, col3 = st.columns([1,1,1])
            if round(probability) == 0:
                col1.write(f':red[**Yes, this is a Fake Face**]')
                col1.wrtie(f'with probability of {probability*100:.2f}%.')
                explode = (0.1, 0)

            else:
                col1.write(f':green[**No, this is a Real Face**]')
                explode = (0, 0.1)

            # st.write(f'**{probability}**')

            labels = 'Fake', 'Real'
            sizes = [1-probability, probability]
            colors = ['#d62728','#2ca02c']

            # fig1, ax1 = plt.subplots()
            # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            #         shadow=False, startangle=90)
            # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            # col2.pyplot(fig1)

            fig = px.pie(names= labels,
                         values = sizes,
                         color=labels,
                         color_discrete_map = dict(zip(labels, colors)),
                         hover_data = [labels])
            col2.plotly_chart(fig, use_container_width=True)

        elif model_kind == 'Specific Classifier':
            probs = [model_.inference(my_upload) for model_ in models]
            probs_dict = dict(zip(model_names, probs))

            # st.write(probs_dict)
            # st.table(probs_dict)
            df_ = pd.DataFrame({ 'Model Name' : model_names,
                                    'Probability' : (1 - np.array(probs))*100})
            st.dataframe(df_.style.format({'Probability': '{:.2f}'}))




    st.divider()

    # if 'pipe' not in st.session_state:
    #     st.session_state['pipe'] = model_pipline(model=model_name)


if __name__ == "__main__":
    main()
