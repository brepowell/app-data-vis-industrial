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


with st.expander('Data Exploration'):

  # Find columns where all values are the same
  constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
  num_total = df.shape[1]
  num_constant = len(constant_cols)
  percent_useless = (num_constant / num_total) * 100
  
  st.write(f"Data Integrity Report:")
  st.write(f"- Total Features: {num_total}")
  st.write(f"- Zero-Variance Features to Drop: {num_constant} ({percent_useless:.2f}%)")
    
  df_clean = df.drop(columns=constant_cols)
  st.write(f"Dropped {len(constant_cols)} constant features.")

  missing_data = df.isnull().mean() * 100
  # Filter for features with more than 50% missing data
  high_missing = missing_data[missing_data > 50]
  st.write(f"Number of features with >50% missing values: {len(high_missing)}")

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='Feature_0', y='Feature_1', color="Label")





