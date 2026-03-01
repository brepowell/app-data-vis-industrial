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
  st.write("There are 28 features that have over 50% missing values. Of those, some features, like 'Feature_72' and 'Feature_73', have a larger difference between passes and fails. That could be significant.
  comparison_df

# Automatically identify "High Signal" columns (Difference > 5%)
high_signal_cols = comparison_df[comparison_df['Difference_%'] > 5].index

# Create new indicator columns and then drop original high-missing columns
for col in high_signal_cols:
    df[f'{col}_is_missing'] = df[col].isnull().astype(int)

# Drop ALL columns with > 50% missingness
df = df.drop(columns=high_missing_cols)

with st.expander('Missingness Indicator Columns'):
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

with st.expander('Failure Rate Per Day'):

  st.write("Sunday has a slightly higher rate of failure.")
  
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
  st.pyplot(plt)


with st.expander('Days vs. Hours - Probability of Failure'):

  st.write("There may be a trend at the beginning of shifts at 8AM.")
  # Extract hour
  df['Hour'] = df['Timestamp'].dt.hour
  
  # Create a pivot table for the heatmap
  heatmap_data = df.pivot_table(index='Day_of_Week', columns='Hour', values='is_fail', aggfunc='mean')
  heatmap_data = heatmap_data.reindex(day_order)
  
  plt.figure(figsize=(12, 6))
  sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False)
  plt.title('Heatmap: Failure Probability by Hour and Day')
  st.pyplot(plt)

with st.expander('Rolling Failure Rate - Peaks of Failure Over Time'):
  # Sort by time
  df = df.sort_values('Timestamp')
  
  # Calculate a rolling failure rate (e.g., over the last 50 units processed)
  df['Rolling_Fail_Rate'] = df['is_fail'].rolling(window=50).mean() * 100

  # Calculate the overall mean and standard deviation of the rolling rate
  avg_rate = df['Rolling_Fail_Rate'].mean()
  std_rate = df['Rolling_Fail_Rate'].std()

  # Plot
  plt.figure(figsize=(14, 6))
  plt.plot(df['Timestamp'], df['Rolling_Fail_Rate'], color='red', linewidth=1)
  plt.fill_between(df['Timestamp'], df['Rolling_Fail_Rate'], color='red', alpha=0.1)
  plt.title('50-Unit Rolling Failure Rate Over Time')
  plt.ylabel('Failure Rate (%)')
  plt.xlabel('Date')

  # Add 'Control Limits' to the plot
  plt.axhline(y=avg_rate, color='green', linestyle='--', label='Average')
  plt.axhline(y=avg_rate + (2 * std_rate), color='orange', linestyle='--', label='Warning Limit')
  plt.axhline(y=avg_rate + (3 * std_rate), color='red', linestyle='--', label='Action Limit')
  plt.legend()
  
  st.pyplot(plt)

#################
# MORE FEATURES #
#################
st.write("### Anomaly Detection:")

# Setup Time Filters
df['Day'] = df['Timestamp'].dt.day_name()
df['Hour'] = df['Timestamp'].dt.hour

# Define our "Spike" criteria: Wed/Sat at 8 AM
is_spike = ((df['Day'] == 'Wednesday') | (df['Day'] == 'Saturday')) & (df['Hour'] == 8)

# Split the data into "Spike Group" and "Normal Group"
# We only look at features (excluding Timestamp, Label, and Time info)
features_only = [col for col in df.columns if 'Feature_' in col and '_is_missing' not in col]

spike_group = df[is_spike][features_only]
normal_group = df[~is_spike][features_only]

# Calculate the "Z-Score" or Deviation
# We want to see which features moved the most relative to their normal standard deviation
spike_means = spike_group.mean()
normal_means = normal_group.mean()
normal_std = normal_group.std()

# Calculate the 'Signal Shift'
# Formula: (Mean during Spike - Mean during Normal) / Standard Deviation
# This tells us how many "standard deviations" the sensor moved
shifts = (spike_means - normal_means) / normal_std

# Get the Top 5 most "deviant" sensors (absolute value)
top_5_sensors = shifts.abs().sort_values(ascending=False).head(5)

st.write("--- Top 5 Sensors Experiencing Anomalies during Wed/Sat 8 AM Spikes ---")
for sensor, shift_val in top_5_sensors.items():
    direction = "HIGHER" if shifts[sensor] > 0 else "LOWER"
    st.write(f"{sensor}: Shifted {abs(shift_val):.2f} standard deviations {direction} than normal.")
  
#################
# MORE FEATURES #
#################

st.write("### Other Features:")

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='Feature_0', y='Feature_1', color="Label")

with st.expander('X and Y'):
  st.write('**X**')
  X = df_clean.drop('Label', axis=1)
  X

  st.write('**y**')
  y = df_clean.Label
  y

