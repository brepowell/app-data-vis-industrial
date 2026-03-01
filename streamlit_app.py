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

st.write('## Data Exploration') 

##########################
# ZERO-VARIANCE FEATURES #
##########################

st.write(f"Data Integrity Report:")
# Find columns where all values are the same

constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
num_total = df.shape[1]
num_constant = len(constant_cols)
percent_useless = (num_constant / num_total) * 100

st.write(f"- Total Features: {num_total}")
st.write(f"- Zero-Variance Features to Drop: {num_constant} ({percent_useless:.2f}%)")
  
df_clean = df.drop(columns=constant_cols)
st.write(f"Dropped {len(constant_cols)} constant features.")

###############
# MISSINGNESS #
###############
st.write("Data Missingness Report:")

# 1. Calculate missingness globally and filter columns
missing_pct = df.isnull().mean() * 100
high_missing_cols = missing_pct[missing_pct > 50].index

# 2. Group by Label and calculate missingness for those specific columns
# We take the mean of the null check to get the percentage
comparison_df = df[high_missing_cols].isnull().groupby(df['Label']).mean().T * 100

# 3. Rename columns and calculate the difference
comparison_df.columns = ['Missing_in_Passes_%', 'Missing_in_Fails_%']
comparison_df['Difference'] = (comparison_df['Missing_in_Fails_%'] - comparison_df['Missing_in_Passes_%']).abs()

# 4. Sort and display
comparison_df = comparison_df.sort_values(by='Difference', ascending=False).round(2)
print(comparison_df)

st.write(f"- Number of features with >50% missing values: {len(high_missing_cols)}")

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='Feature_0', y='Feature_1', color="Label")



