import streamlit as st
import pandas as pd

st.title('⚡SECOM Industrial Dataset Visualization')

st.write('This app allows you to explore the SECOM dataset')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/brepowell/app-data-vis-industrial/refs/heads/master/secom_combined.csv")
  df

  st.write('**X**')
  X = df.drop('Label', axis=1)
  X

  st.write('**y**')
  y = df.Label
  y
