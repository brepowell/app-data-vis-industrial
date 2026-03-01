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

st.write("### Zero-Variance Features Report:")
st.write("These are features that do not change in the dataset. They are not useful in determining the whether the manufacturing passes or fails.")
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
num_total = df.shape[1]
num_constant = len(constant_cols)
percent_useless = (num_constant / num_total) * 100

st.write(f"- Total Features: {num_total}")
st.write(f"- Zero-Variance Features to Drop: {num_constant} ({percent_useless:.2f}%)")
  
df_clean = df.drop(columns=constant_cols)

###############
# MISSINGNESS #
###############
st.write("### Data Missingness Report:")
st.write("Data that is missing from pass or fail states might be important features, so I do not want to delete them before exploring to see if they are important.")
st.write("In the table below, I look at the percent of missing values in the Passes vs. the Fails to see whether or not the features are important to judging the fails.")
st.write("Low Difference_% means the data is safe to drop because the data is missing across both Passes and Fails.")
st.write("High Difference_% means the data may be important.")

# 1. Calculate missingness globally
missing_pct = df.isnull().mean() * 100
high_missing_cols = missing_pct[missing_pct > 50].index
st.write(f"- Number of features with >50% missing values: {len(high_missing_cols)}")

# 2. Group by Label and calculate missingness for those specific columns
# Take the mean of the null check to get the percentage
comparison_df = df[high_missing_cols].isnull().groupby(df['Label']).mean().T * 100

# 3. Rename columns and calculate the difference
comparison_df.columns = ['Missing_in_Passes_%', 'Missing_in_Fails_%']
comparison_df['Difference_%'] = (comparison_df['Missing_in_Fails_%'] - comparison_df['Missing_in_Passes_%']).abs()

# 4. Sort and display
comparison_df = comparison_df.sort_values(by='Difference_%', ascending=False).round(2)
comparison_df

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='Feature_0', y='Feature_1', color="Label")



