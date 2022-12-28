import streamlit as st
import collections
from collections import OrderedDict
from matplotlib import cm
import pylab
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow import  keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

st.sidebar.title("Created By:")
st.sidebar.markdown("Isha Indhu S")
st.sidebar.subheader("To access the complete project: ")
st.sidebar.markdown("https://github.com/IshaIndhu/DL-Binary-Classification-Higgs-Boson")

def pre_processing(df):
	le = preprocessing.LabelEncoder()
	le.fit(df['Label'])
	Labels = le.transform(df['Label'])
	imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')   
	df1 = df.drop(['Label','EventId'],inplace = False,axis = 1)
	data = imp_mean.fit_transform(df1)
	df_updated = pd.DataFrame(data,columns = df1.columns)
	df_updated.drop(['DER_mass_MMC', 'DER_sum_pt'],inplace=True,axis=1)

	cols = df_updated.columns
	scaler = StandardScaler()
	df_updated = pd.DataFrame(scaler.fit_transform(df_updated),columns=df_updated.columns)

	X_norm = MinMaxScaler().fit_transform(df_updated)
	chi_selector = SelectKBest(chi2, k=10)
	chi_selector.fit(X_norm, Labels)
	selected_features = chi_selector.get_support(indices=True)
	cols = df_updated.columns.values.tolist()
	s = []
	for i in selected_features:
	    s.append(df_updated.columns[i])
	df_optimized = df_updated[s]
	return df_optimized,Labels

def base_model():
    model = Sequential()
    model.add(Dense(32, input_dim=10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='SparseCategoricalCrossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


st.title('TMLC Deep Learning Project - 1')
st.header(' Binary Classification using Higgs Boson dataset')

st.subheader('Select the method of input:')
option=st.radio('Radio', ["Upload the Dataset"])

df = pd.DataFrame()
model = base_model()

if option == "Upload the Dataset":
	f = st.file_uploader("Dataset", type={"csv"})
	if f is not None:
		df = pd.read_csv(f)
		st.write(df.head())


		st.subheader("Correlation plot")
		fig, ax = plt.subplots(figsize=(15,15))
		sns.heatmap(df.corr(), cmap = 'Reds',annot=True,linewidths=.7, ax=ax)
		st.pyplot(plt)	    


		#st.subheader("Choose train test split(default = 0.2): ")
		k = st.number_input('Choose train test split(default = 20): ', min_value=20, max_value=100)

		X,y = pre_processing(df)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=k, random_state=42)

		

		history = model.fit(X_train, y_train,
		                    validation_data=(X_test, y_test),
		                    epochs=20,verbose=2,shuffle=True)

		prediction = model.predict(x=X_test,batch_size=10,verbose = 0)

		from sklearn.metrics import classification_report,confusion_matrix
		y_pred = np.argmax(prediction,axis = -1)

		st.subheader("CLASSIFICATION REPORT: ")
		st.table(classification_report(y_test,y_pred, target_names = ["b","s"], output_dict=True))
		st.subheader("The Accuracy obtained from the test data: ")
		st.write(accuracy_score(y_test, y_pred))
