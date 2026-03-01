import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('⚡SECOM Industrial Dataset Visualization')

st.write('This app allows you to explore the SECOM dataset')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/brepowell/app-data-vis-industrial/refs/heads/master/secom_combined.csv")
  df

st.write('## Data Exploration') 

num_total = df.shape[1]
st.write(f"- Total Features: {num_total}")

##########################
# ZERO-VARIANCE FEATURES #
##########################

st.write("### Zero-Variance Features:")
st.write("These are features that do not change in the dataset. They are not useful in determining the whether the manufacturing passes or fails.")
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
num_constant = len(constant_cols)
percent_useless = (num_constant / num_total) * 100

st.write(f"- Zero-Variance Features to Drop: {num_constant} ({percent_useless:.2f}%)")
  
df_clean = df.drop(columns=constant_cols)

###############
# MISSINGNESS #
###############
st.write("### Data Missingness Report:")
st.write("Data that is missing from pass or fail states might be important features, so I do not want to delete them before exploring to see if they are important.")
st.write("In the table below, I look at the percent of missing values in the Passes vs. the Fails to see whether or not the features are important to judging the fails.")
st.write("- Low Difference_% means the data is safe to drop because the data is missing across both Passes and Fails.")
st.write("- High Difference_% means the data may be important.")

# Calculate missingness globally and group by label
missing_pct = df.isnull().mean() * 100
high_missing_cols = missing_pct[missing_pct > 50].index
st.write(f"- Number of features with >50% missing values: {len(high_missing_cols)}")

# Group by Label and calculate missingness for those specific columns
# Take the mean of the null check to get the percentage
comparison_df = df[high_missing_cols].isnull().groupby(df['Label']).mean().T * 100

# Rename columns and calculate the difference
comparison_df.columns = ['Missing_in_Passes_%', 'Missing_in_Fails_%']
comparison_df['Difference_%'] = (comparison_df['Missing_in_Fails_%'] - comparison_df['Missing_in_Passes_%']).abs()

# Sort and display
comparison_df = comparison_df.sort_values(by='Difference_%', ascending=False).round(2)

with st.expander('Missingness across Passes and Fails'):
  comparison_df

# Automatically identify "High Signal" columns (Difference > 5%)
high_signal_cols = comparison_df[comparison_df['Difference_%'] > 5].index

# Create new indicator columns and then drop original high-missing columns
for col in high_signal_cols:
    df[f'{col}_is_missing'] = df[col].isnull().astype(int)

# Drop ALL columns with > 50% missingness
df = df.drop(columns=high_missing_cols)

st.write(f"Created {len(high_signal_cols)} indicator columns and dropped {len(high_missing_cols)} original features.")

# Filter the dataframe for the Label and new indicator columns
indicator_cols = [col for col in df.columns if '_is_missing' in col]
plot_data = df[['Label'] + indicator_cols]

# Calculate the Correlation Matrix
corr_matrix = plot_data.corr()

# Plotting
st.write("There does not seem to be a strong linear correlation between missingness and the label of Pass vs. Fail. There might be a nonlinear relationship not yet explored. ")
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix[['Label']].sort_values(by='Label', ascending=False), 
            annot=True, 
            cmap='coolwarm', 
            center=0)

plt.title("Correlation: Missingness Indicators vs. Label")
st.pyplot(plt)

###############
# TIME SERIES #
###############
st.write("### Time-Relevant Features:")

# Convert timestamp to actual datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)

# Extract day of the week
df['Day_of_Week'] = df['Timestamp'].dt.day_name()

# Calculate failure rate per day
# Mapping -1 to 0 and 1 to 1 makes calculating the 'mean' the same as 'failure rate'
df['is_fail'] = df['Label'].map({-1: 0, 1: 1})
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
fail_rate_by_day = df.groupby('Day_of_Week')['is_fail'].mean() * 100

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=fail_rate_by_day.index, y=fail_rate_by_day.values, order=day_order, palette='viridis')
plt.title('Percentage of Failures by Day of the Week')
plt.ylabel('Failure Rate (%)')
plt.xlabel('Day')
plt.show()

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='Feature_0', y='Feature_1', color="Label")

with st.expander('X and Y'):
  st.write('**X**')
  X = df_clean.drop('Label', axis=1)
  X

  st.write('**y**')
  y = df_clean.Label
  y

