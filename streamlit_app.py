import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('⚡SECOM Industrial Dataset Visualization')

st.write('This app allows you to explore the SECOM dataset')

with st.expander('Raw Data'):
    st.write('**Raw Data**')
    df = pd.read_csv("https://raw.githubusercontent.com/brepowell/app-data-vis-industrial/refs/heads/master/secom_combined.csv")
    st.dataframe(df) # Displays the original data in an interactive table

    # Calculate the descriptive statistics
    df_description = df.describe()

    # Display the descriptive statistics
    st.write("Descriptive Statistics (df.describe())")
    st.dataframe(df_description)

st.write('## Data Exploration') 

num_total = df.shape[1]
st.write(f"- Total Features: {num_total}")


###############
# TIME SERIES #
###############
st.write("### Raw Time Features:")

# Convert timestamp to actual datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)

# Extract day of the week
df['Day_of_Week'] = df['Timestamp'].dt.day_name()

# Group by day and count the number of rows
logs_per_day = df.groupby(df['Timestamp'].dt.date).size()

with st.expander('Sensor Logs Per Day'):
   
    st.write("This shows that the number of logs per day is not consistant over time. To even this out, I will flatten it out to one average reading per hour.")

    # Plot line chart
    plt.figure(figsize=(12, 5))
    plt.plot(logs_per_day.index, logs_per_day.values, color='skyblue', linewidth=2)
    plt.fill_between(logs_per_day.index, logs_per_day.values, color='skyblue', alpha=0.3)
    plt.title("Number of Sensor Logs Per Day")
    st.pyplot(plt)

    # Check for consistency
    mean_logs = logs_per_day.mean()
    std_logs = logs_per_day.std()
    st.write(f"Average logs per day: {mean_logs:.2f} (±{std_logs:.2f})")

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

with st.expander('Rolling Failure + Logs per day'):
   
    # 3. Calculate Log Density (Logs per Day)
    # We resample to 'D' (Day) and count the rows
    log_density = df.resample('D', on='Timestamp').size()

    # 4. Create the Dual-Axis Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Axis 1: Rolling Failure Rate (Red Line) ---
    color_fail = '#e63946' # Soft Red
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Rolling Failure Rate (Window=100)', color=color_fail, fontweight='bold')
    ax1.plot(df['Timestamp'], df['Rolling_Fail_Rate'], color=color_fail, linewidth=2, label='Failure Rate')
    ax1.tick_params(axis='y', labelcolor=color_fail)
    ax1.grid(True, alpha=0.3)

    # --- Axis 2: Log Density (Blue Line) ---
    ax2 = ax1.twinx() 
    color_logs = '#457b9d' # Steel Blue
    ax2.set_ylabel('Logs Per Day (Density)', color=color_logs, fontweight='bold')
    # Note: log_density index is the date, values is the count
    ax2.plot(log_density.index, log_density.values, color=color_logs, linewidth=2, linestyle='--', label='Log Count')
    ax2.tick_params(axis='y', labelcolor=color_logs)

    # 5. Final Formatting
    plt.title("Comparison: Failure Rate vs. Data Collection Density", fontsize=14)
    fig.tight_layout()

    # Display in Streamlit
    st.pyplot(fig)

###############
# MISSINGNESS #
###############

st.write("### Data Missingness Report:")

# Define high_missing_cols 
null_percent = df.isnull().mean()
high_missing_cols = null_percent[null_percent > 0.5].index.tolist()

with st.expander('All Missingness'):

    # Calculate the null values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    top_missing_col = df.isnull().sum().idxmax()
    col1, col2 = st.columns(2)
    col1.metric("Total Missing Values", total_nulls)
    col2.metric("Top Missing Column", top_missing_col)

    null_counts = null_counts.sort_values(ascending=False)

    st.write("**Missing Values per Column**")
    st.table(null_counts)

# Ensure high_missing_cols isn't empty to avoid errors
if high_missing_cols:

    # Group by Label and calculate missingness for those specific columns
    # Take the mean of the null check to get the percentage
    comparison_df = df[high_missing_cols].isnull().groupby(df['Label']).mean().T * 100
    
    # Rename columns and calculate the difference
    comparison_df.columns = ['Missing_in_Passes_%', 'Missing_in_Fails_%']
    comparison_df['Difference_%'] = (comparison_df['Missing_in_Fails_%'] - comparison_df['Missing_in_Passes_%']).abs()
    
    # Sort and display
    comparison_df = comparison_df.sort_values(by='Difference_%', ascending=False).round(2)

    # Define high_signal_cols
    high_signal_cols = comparison_df[comparison_df['Difference_%'] > 5].index

    # Create new indicator columns and then drop original high-missing columns
    for col in high_signal_cols:
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)
    
    # Drop ALL columns with > 50% missingness
    df = df.drop(columns=high_missing_cols)
else:
    st.write("No columns found with >50% missing values.")

with st.expander('Missingness across Passes and Fails'):
        
    st.write("Data that is missing from pass or fail states might be important features because there may be a correlation between a sensor failing and a failed component, so I do not want to delete them before exploring to see if they are important.")
    st.write("In the table below, I look at the percent of missing values in the Passes vs. the Fails to see whether or not the features are important to judging the fails.")
    st.write("- Low Difference_% means the data is probably safe to drop because the data is missing across both Passes and Fails.")
    st.write("- High Difference_% means the data may be important.")
    st.write(f"- Number of features with >50% missing values: {len(high_missing_cols)}")
    st.write("There are 28 features that have over 50 percent missing values. Of those, some features, like Feature_72 and Feature_73, have a larger difference between passes and fails. That could be significant.")
    comparison_df

with st.expander('Missingness Indicator Columns'):
  st.write("I will make some Shadow Variables to see if the missing data is significant to predicting failures.")
  
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


#####################
# RESAMPLE FEATURES #
#####################

df_indexed = df.set_index('Timestamp')

# Define how to handle each column type
agg_rules = {}
for col in df_indexed.columns:
    if col == 'Label' or '_is_missing' in col:
        agg_rules[col] = 'max'  # Keep the failure/missing signal
    elif df_indexed[col].dtype in ['float64', 'int64']:
        agg_rules[col] = 'mean' # Only average numeric sensor readings
    # We ignore string columns like 'Day_of_Week' as they are recreated later

# Create the Hourly Dataframe
df_hourly = df_indexed.resample('1H').agg(agg_rules)
df_hourly = df_hourly.dropna(subset=['Label'])

st.write(f"Due to the fact that the number of logs per day is uneven, I resampled the data to just a mean per hour per sensor.")
st.write(f"Reduced from {len(df)} logs to {len(df_hourly)} hours.")

##########################
# ZERO-VARIANCE FEATURES #
##########################

st.write("### Zero-Variance Features:")
st.write("These are features that do not change in the dataset. They are not useful in determining the whether the manufacturing passes or fails.")

# Use only numeric columns for nunique check to be safe
numeric_cols = df_hourly.select_dtypes(include=['number']).columns
constant_cols = [col for col in numeric_cols if df_hourly[col].nunique() <= 1]
num_constant = len(constant_cols)
percent_useless = (num_constant / num_total) * 100

st.write(f"- Zero-Variance Features to Drop: {num_constant} ({percent_useless:.2f}%)")
df_hourly = df_hourly.drop(columns=constant_cols)

with st.expander('Resampling Validation: Raw vs. Hourly Failure Heatmaps'):
    
    # Prepare Data for both DataFrames
    # Ensure Day_of_Week and Hour exist for both
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for d in [df, df_hourly]:
        ts_source = d.index if isinstance(d.index, pd.DatetimeIndex) else d['Timestamp']
        d['Day_of_Week'] = ts_source.day_name()
        d['Hour'] = ts_source.hour

    # Create the Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # --- Heatmap 1: Raw Data (Log-Density Probability) ---
    raw_pivot = df.pivot_table(index='Day_of_Week', columns='Hour', values='Label', aggfunc='mean')
    raw_pivot = raw_pivot.reindex(day_order)
    sns.heatmap(raw_pivot, cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Fail Rate per Log'})
    ax1.set_title('Raw Data: Failure Probability (Biased by Log Count)')

    # --- Heatmap 2: Hourly Data (Time-Density Probability) ---
    hourly_pivot = df_hourly.pivot_table(index='Day_of_Week', columns='Hour', values='Label', aggfunc='mean')
    hourly_pivot = hourly_pivot.reindex(day_order)
    sns.heatmap(hourly_pivot, cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Fail Rate per Hour'})
    ax2.set_title('Hourly Resampled: Failure Probability (Equitable Time)')

    plt.tight_layout()
    st.pyplot(fig)

#################
# MORE FEATURES #
#################
st.write("### Anomaly Detection:")

# Access the index since Timestamp isn't a column anymore
df_hourly['Day'] = df_hourly.index.day_name()
df_hourly['Hour'] = df_hourly.index.hour

# Define "Spike" criteria: Wed/Sat at 8 AM
is_spike = ((df_hourly['Day'] == 'Wednesday') | (df_hourly['Day'] == 'Saturday')) & (df_hourly['Hour'] == 8)

# Split the data into "Spike Group" and "Normal Group"
# We only look at features (excluding Timestamp, Label, and Time info)
features_only = [col for col in df_hourly.columns if 'Feature_' in col and '_is_missing' not in col]

spike_group = df_hourly[is_spike][features_only]
normal_group = df_hourly[~is_spike][features_only]

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

st.write("### PCA and Dimensionality Reduction:")

from sklearn.ensemble import RandomForestClassifier

# Drop both Label and Timestamp/Time (using errors='ignore' in case names vary)
columns_to_drop = ['Label', 'Timestamp', 'Day_of_Week', 'Day', 'Rolling_Fail_Rate']
X = df_hourly.drop(columns=columns_to_drop, errors='ignore')

# Ensure we only have numeric data for the mean calculation and the model
X = X.select_dtypes(include=['number'])

# Fill missing values with the mean of the numeric columns
X = X.fillna(X.mean())

y = df_hourly['Label']

# Train a quick Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get importance and filter for the indicator columns
importances = pd.Series(rf.feature_importances_, index=X.columns)
indicator_importance = importances[importances.index.str.contains('_is_missing')].sort_values(ascending=False)

st.write("Feature Importance of Missingness Indicators")
st.bar_chart(indicator_importance)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA and start by looking at 95% of the variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Report Results
n_components = pca.n_components_
st.write(f"Feature Count Before PCA: {X.shape[1]}")
st.write(f"Reduced to {n_components} components while keeping 95% of variance.")

with st.expander('Interactive Data Visualization'):
    st.write("Select sensors to visualize how they relate to manufacturing failures.")
    
    # Only allow numeric sensor features for plotting
    plot_options = [col for col in df_hourly.columns if 'Feature_' in col]
    
    if len(plot_options) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox('Select X-axis sensor:', plot_options, index=0)
        with col2:
            y_axis = st.selectbox('Select Y-axis sensor:', plot_options, index=1)
        
        # Streamlit's native scatter chart
        st.scatter_chart(
            data=df_hourly, 
            x=x_axis, 
            y=y_axis, 
            color="Label", 
            use_container_width=True
        )
    else:
        st.warning("Not enough numeric features remaining after cleaning to create a scatter plot.")


df_hourly
