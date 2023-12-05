import streamlit as st
from PIL import Image
import model_pipeline
import torch
#%%


def sidebar():
    with st.sidebar:
        model_kind = st.radio(
            "Model Kind : ",
            ["Broad Classifer", "Specific Classifier"],
            captions=["Classifies fake and real human faces from varity of models.",
                      "Classifies fake and real human faces and specifies the most likely model used to generate the image if fake.",],
            index=0,
        )
        model_type = st.radio(
            "Model Type :",
            ['CNN', 'Transformer'],
            captions = ["Uses a Convolutional Neural Network based network to classify",
                        "Uses a Transformer based model to classifer"]
        )
        if model_type == "CNN":
            model_name = "DenseNet"
        elif model_type == "Transformer":
            model_name = "VIT"

        st.write(f"Model in use : **{model_name}**")

    return model_name, model_type

def main():
    st.header("Fake Image Detection")
    st.divider()
    st.subheader('Image File Uploader')
    model_name, model_type = sidebar()
    my_upload = st.file_uploader('Upload an image', type=['png','jpg','jpeg'])

    if my_upload is None:
        st.stop()

    st.image(Image.open(my_upload), caption='Uploaded Image', use_column_width=True)
    X = model_pipeline.preprocess_image(my_upload, model_type)

    clicked = st.button('Is that a Fake Face')

    if clicked:
        torch.set_grad_enabled(False)
        model = model_pipeline.load_model(model_name)
        probability = model_pipeline.model_inference(X, model)
        st.write(f'**{probability}**')


    st.divider()

    # if 'pipe' not in st.session_state:
    #     st.session_state['pipe'] = model_pipline(model=model_name)


if __name__ == "__main__":
    main()
