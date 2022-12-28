import pandas as pd
import numpy as np
import streamlit as st
from keras.models import load_model
from tensorflow import keras
import tensorflow as tf

from prediction import get_prediction


st.set_page_config(page_title='Higgs Boson Event Detection', page_icon="🌎", layout="wide", initial_sidebar_state='expanded')

model = load_model('keras_model.h5')

# creating option list for dropdown menu

features = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis','PRI_tau_pt', 'Weight']

st.markdown("<h1 style='text-align: center;'>Higgs Boson Event Detection 🌎 </h1>", unsafe_allow_html=True)


def main():
    with st.form('prediction_form'):

        st.header("Predict the input for following features:") 


    
        
        
        DER_met_phi_centrality  = st.slider('DER_met_phi_centrality', 0.0, 1.414, value=0.0, format="%f")
        DER_mass_transverse_met_lep = st.slider('DER_mass_transverse_met_lep', 0.0, 690.075, value=0.0, format="%f")
        DER_lep_eta_centrality  = st.slider('DER_lep_eta_centrality', 0.0, 1.000, value=0.0, format="%f")
        PRI_jet_num = st.slider('PRI_jet_num', 0.0, 3.000, value=0.0, format="%f")
        
        Weight= st.selectbox( 'Weight:', [1.877474,1.681611,0.018636,2.497259,4.505083,6.245333,7.8222])
        submit = st.form_submit_button("Predict")

    if submit:

        data= np.array([DER_met_phi_centrality, DER_mass_transverse_met_lep, DER_lep_eta_centrality, PRI_jet_num, Weight]).reshape(1, -1)
        
        pred = get_prediction(data=data, model=model)

        
        if pred [0][0] < 0.5:
            Label = 'Signal'
        elif pred [0][0] > 0.5:
            Label = 'Background'

        
        st.write(f"The predicted Higgs Boson Event Detection is: {Label}")



if __name__ == '__main__':
    main()