import streamlit as st
from PIL import Image
import model_pipeline as mp
import torch
import plotly.express as px
import pandas as pd
import numpy as np
# from scipy.special import softmax
#%%
import matplotlib.pyplot as plt


def sidebar():
    with st.sidebar:
        # model_kind = st.radio(
        #     "Model Kind : ",
        #     ["Broad Classifier", "Specific Classifier"],
        #     captions=["Classifies fake and real human faces from varity of models.",
        #               "Classifies fake and real human faces and specifies the most likely model used to generate the image if fake.",],
        #     index=0,
        # )
        model_type_class = st.selectbox(
            "Image Classifier Type :",
            ['CNN', 'Transformer'],
            # captions = ["Uses a Convolutional Neural Network based network to classify",
            #             "Uses a Transformer based model to classifer"],
            index = 0,
            key = hash('model_type_class')
        )
        if model_type_class == 'CNN':
            model_name_class = 'DenseNet'
        elif model_type_class == 'Transformer':
            model_name_class = 'VIT'

        model_name_spec = st.toggle('Parent Classifier', value=False)
        if model_name_spec:
            model_type_spec = st.selectbox(
                "Specific Classifier Type:",
                ['CNN', 'Transformer'],
                # captions = ["Uses a Convolutional Neural Network based network to classify",
                #             "Uses a Transformer based model to classifer"],
                index=0,
                key=hash('model_type_spec')
            )
            if model_type_spec == 'CNN':
                model_name_spec = 'DenseNet'
            elif model_type_spec == 'Transformer':
                model_name_spec = 'VIT'

    return model_name_spec, model_name_class



def main():
    st.image(f'logo/black_logo.png')
    # st.header("Face Auth")
    st.divider()
    st.subheader('Image File Uploader')

    model_name_spec, model_name_class = sidebar()

    my_upload = st.file_uploader('Upload an image', type=['png','jpg','jpeg'])



    # if model_kind == 'Broad Classifier':
    #     if model_name == 'DenseNet':
    #         model = mp.DenseNet_pipeline(model_name)
    #     if model_name == 'VIT':
    #         model = mp.VIT_pipeline(model_name)
    # elif model_kind == 'Specific Classifier':
    #     model_subtypes = ['diffusion', 'GAN', 'GANprintR']
    #     model_names = [f'{model_name}_{subtype}' for subtype in model_subtypes]
    #     if model_name == 'DenseNet':
    #         models = [mp.DenseNet_pipeline(name) for name in model_names]
    #     if model_name == 'VIT':
    #         models = [mp.VIT_pipeline(name) for name in model_names]

    if model_name_class == 'DenseNet':
        classifier_model = mp.DenseNet_pipeline(model_name_class)
    elif model_name_class == 'VIT':
        classifier_model = mp.VIT_pipeline(model_name_class)
    else: print(f'Skipped If loop {model_name_class}')

    if model_name_spec:
        model_subtypes = ['diffusion', 'GAN', 'GANprintR']
        model_names = [f'{model_name_spec}_{subtype}' for subtype in model_subtypes]
        if model_name_spec == 'DenseNet':
            models = [mp.DenseNet_pipeline(name) for name in model_names]
        if model_name_spec == 'VIT':
            models = [mp.VIT_pipeline(name) for name in model_names]

    if my_upload is None:
        st.stop()

    st.image(Image.open(my_upload), caption='Uploaded Image', use_column_width=True)

    st.write(f"Model in use : **{model_name_class}**")
    clicked = st.button('Is that a Fake Face?')
    st.divider()

    if clicked:
        st.subheader('Classifier Output')
        st.write(f"Model used : **{model_name_class}**")
        torch.set_grad_enabled(False)

        class_prob = classifier_model.inference(my_upload)

        col1, col2, col3 = st.columns([1,1,1])
        if round(class_prob) == 0:
            col1.write(f':red[**Yes, this is a Fake Face**]')
            # col1.write(f'with probability of {class_prob*100:.2f}%.')

        else:
            col1.write(f':green[**No, this is a Real Face**]')

        # st.write(f'**{probability}**')

        labels = 'Fake', 'Real'
        sizes = [1-class_prob, class_prob]
        colors = ['#d62728','#2ca02c']

        fig = px.pie(names= labels,
                     values = sizes,
                     color=labels,
                     color_discrete_map = dict(zip(labels, colors)),
                     hover_data = [labels])
        col2.plotly_chart(fig, use_container_width=True)
        st.divider()

        if 'override_spec' not in st.session_state:
            st.session_state['override_spec'] = False
        def change_override_spec():
            if st.session_state.override_spec:
                st.session_state['override_spec'] = False
            else:
                st.session_state['override_spec'] = True



        if model_name_spec:
            st.subheader('Parent Classifier Output')
            st.write(f"Models used : **{model_names}**")

            if round(class_prob) == 0 or st.session_state['override_spec']:
                if st.session_state['override_spec']:
                    st.write('*Showing because of override*')
                st.write('This image is from: ')
                probs = [model_.inference(my_upload) for model_ in models]
                probs = np.array(probs)
                probs = (1 - probs)/np.sum(1 - probs)
                # probs_dict = dict(zip(model_subtypes, probs))

                # st.write(probs_dict)
                # st.table(probs_dict)
                df = pd.DataFrame({'Parent': model_subtypes,
                                    'Probability': np.array(probs) * 100})
                df_ = df.style.format({'Probability': '{:.2f}'})
                most_likely_df = df.max()
                st.dataframe(df_)
                # st.write(f'The image is most likely from a **{most_likely_df.iloc[0]}** based model with a probability of **{most_likely_df.iloc[1]:.2f}%**')




            else:
                st.write('Since it is a Real Face, not showing Parent Classifier')
                override_spec = st.button('Override',
                                          on_click=change_override_spec)


    # if 'pipe' not in st.session_state:
    #     st.session_state['pipe'] = model_pipline(model=model_name)


if __name__ == "__main__":
    main()
