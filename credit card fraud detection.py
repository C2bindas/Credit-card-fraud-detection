%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import numpy as np
from fasteda import fast_eda
from datacleaner import autoclean
import scipy
import scipy.stats as stats
from collections import Counter

df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.head()
df.tail()
df.shape
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
df.describe()
df.skew()
df.kurtosis()
df.duplicated().sum()



df = df.drop_duplicates()
df.duplicated().sum()


df.drop("Time", axis = 1, inplace = True)
df.head()
Outlier Analysis
feature_list = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
'V10', 'V11',
'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
'V20', 'V21',
'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
'Amount']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(11, 17))
fig.suptitle('Features vs Class\n', size=18)

features_to_visualize = ['V17', 'V10', 'V12', 'V16', 'V14', 'V3', 'V7',
'V11', 'V4']

for i, feature in enumerate(features_to_visualize):
row, col = i // 3, i % 3 # Calculate the row and column for the
subplot

sns.boxplot(ax=axes[row, col], data=df, x='Class', y=feature,
palette='viridis')
axes[row, col].set_title(f"{feature} Distribution")

plt.tight_layout()

plt.show()


def IQR_method(df, n, features):

outlier_list = []
for column in features:

Q1 = np.percentile(df[column], 25)

Q3 = np.percentile(df[column], 75)

IQR = Q3 - Q1

outlier_step = 1.5 * IQR

outlier_list_column = df[(df[column] < Q1 - outlier_step) |
(df[column] > Q3 + outlier_step)].index

outlier_list.extend(outlier_list_column)

outlier_count = Counter(outlier_list)

multiple_outliers = [k for k, v in outlier_count.items() if v > n]

total_outliers = len(multiple_outliers)
print('Total number of outliers is:', total_outliers)
return multiple_outliers


Outliers_IQR = IQR_method(df, 1, feature_list)

df_out = df.drop(Outliers_IQR, axis=0).reset_index(drop=True)



sns.set_palette('viridis')
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 8))
fig.suptitle('Distributions of Most Important Features after Dropping
Outliers using IQR Method\n', size=18)

axes[0, 0].hist(df_out['V17'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 0].set_title("V17 Distribution")
axes[0, 1].hist(df_out['V10'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 1].set_title("V10 Distribution")
axes[0, 2].hist(df_out['V12'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 2].set_title("V12 Distribution")
axes[1, 0].hist(df_out['V16'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 0].set_title("V16 Distribution")
axes[1, 1].hist(df_out['V14'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 1].set_title("V14 Distribution")
axes[1, 2].hist(df_out['V3'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 2].set_title("V3 Distribution")
axes[2, 0].hist(df_out['V7'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 0].set_title("V7 Distribution")
axes[2, 1].hist(df_out['V11'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 1].set_title("V11 Distribution")
axes[2, 2].hist(df_out['V4'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 2].set_title("V4 Distribution")

plt.tight_layout()
# Show the plot
plt.show()
In [21]:

sns.set_palette('viridis')

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 8))
fig.suptitle('Distributions of Most Important Features after Dropping
Outliers using IQR Method\n', size=18)


row, col = i // 3, i % 3 # Calculate the row and column for the
subplot

plt.tight_layout()

plt.show()

def StDev_method(df, n, features):



outlier_indices = []
for column in features:

data_mean = df[column].mean()
data_std = df[column].std()

cut_off = data_std * 3

outlier_list_column = df[(df[column] < data_mean - cut_off) |
(df[column] > data_mean + cut_off)].index

outlier_indices.extend(outlier_list_column)

outlier_indices = Counter(outlier_indices)
multiple_outliers = [k for k, v in outlier_indices.items() if v >
n]

total_outliers = len(multiple_outliers)
print('Total number of outliers is:', total_outliers)
return multiple_outliers

import seaborn as sns

data_mean, data_std = df['V11'].mean(), df['V11'].std()

cut_off = data_std * 3

lower, upper = data_mean - cut_off, data_mean + cut_off
# Print the lower and upper bound values
print('The lower bound value is:', lower)
print('The upper bound value is:', upper)
# Set the color palette to 'viridis'
sns.set_palette('viridis')
# Create a histogram to visualize the 'V11' feature
plt.figure(figsize=(10, 5))
sns.histplot(x='V11', data=df, bins=70)
# Highlight the regions outside the bounds in red
plt.axvspan(xmin=lower, xmax=df['V11'].min(), alpha=0.2, color='red')
plt.axvspan(xmin=upper, xmax=df['V11'].max(), alpha=0.2, color='red')
# Show the plot
plt.show()



Outliers_StDev = StDev_method(df, 1, feature_list)

df_out2 = df.drop(Outliers_StDev, axis=0).reset_index(drop=True)

sns.set_palette('viridis')

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 8))
fig.suptitle('Distributions of Most Important Features after Dropping
Outliers using Standard Deviation Method\n', size=18)

axes[0, 0].hist(df_out2['V17'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 0].set_title("V17 Distribution")
axes[0, 1].hist(df_out2['V10'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 1].set_title("V10 Distribution")
axes[0, 2].hist(df_out2['V12'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 2].set_title("V12 Distribution")
axes[1, 0].hist(df_out2['V16'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 0].set_title("V16 Distribution")
axes[1, 1].hist(df_out2['V14'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 1].set_title("V14 Distribution")
axes[1, 2].hist(df_out2['V3'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 2].set_title("V3 Distribution")
axes[2, 0].hist(df_out2['V7'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 0].set_title("V7 Distribution")
axes[2, 1].hist(df_out2['V11'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 1].set_title("V11 Distribution")
axes[2, 2].hist(df_out2['V4'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 2].set_title("V4 Distribution")
# Adjust the layout to avoid overlap
plt.tight_layout()
# Show the plot
plt.show()


def z_score_method(df, n, features):

outlier_list = []
threshold = 3 # Z-score threshold for identifying outliers
for column in features:

data_mean = df[column].mean()
data_std = df[column].std()

z_score = abs((df[column] - data_mean) / data_std)

outlier_list_column = df[z_score > threshold].index

outlier_list.extend(outlier_list_column)

outlier_list = Counter(outlier_list)
multiple_outliers = [k for k, v in outlier_list.items() if v > n]

df1 = df[df.index.isin(multiple_outliers)]
total_outliers = df1.shape[0]
print('Total number of outliers is:', total_outliers)
return multiple_outliers


Outliers_z_score = z_score_method(df, 1, feature_list)

df_out3 = df.drop(Outliers_z_score, axis=0).reset_index(drop=True)


sns.set_palette('viridis')

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 8))
fig.suptitle('Distributions of Most Important Features after Dropping
Outliers using Z-score\n', size=18)

axes[0, 0].hist(df_out3['V17'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 0].set_title("V17 Distribution")
axes[0, 1].hist(df_out3['V10'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 1].set_title("V10 Distribution")
axes[0, 2].hist(df_out3['V12'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 2].set_title("V12 Distribution")
axes[1, 0].hist(df_out3['V16'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 0].set_title("V16 Distribution")
axes[1, 1].hist(df_out3['V14'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 1].set_title("V14 Distribution")
axes[1, 2].hist(df_out3['V3'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 2].set_title("V3 Distribution")
axes[2, 0].hist(df_out3['V7'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 0].set_title("V7 Distribution")
axes[2, 1].hist(df_out3['V11'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 1].set_title("V11 Distribution")
axes[2, 2].hist(df_out3['V4'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 2].set_title("V4 Distribution")

plt.tight_layout()

plt.show()


from scipy.stats import median_abs_deviation
def z_scoremod_method(df, n, features):
"""
Identify outliers in a DataFrame using the modified z-score method.
Parameters:
df (DataFrame): The input DataFrame.
n (int): The minimum number of outliers in an observation to be
considered.
features (list): List of feature column names to analyze for
outliers.
Returns:
list: A list of indices corresponding to observations with more than
'n' outliers.
"""
outlier_list = []
threshold = 3
for column in features:

data_mean = df[column].mean()
data_mad = median_abs_deviation(df[column])
mod_z_score = abs(0.6745 * (df[column] - data_mean) / data_mad)

outlier_list_column = df[mod_z_score > threshold].index

outlier_list.extend(outlier_list_column)

outlier_list = Counter(outlier_list)

df1 = df[df.index.isin(multiple_outliers)]
total_outliers = df1.shape[0]
print('Total number of outliers is:', total_outliers)
return multiple_outliers


Outliers_z_score = z_scoremod_method(df, 1, feature_list)

df_out4 = df.drop(Outliers_z_score, axis=0).reset_index(drop=True)
Total number of outliers is: 64564


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 8))
fig.suptitle('Distributions of Most Important Features after Dropping
Outliers using Modified Z-score\n', size=18)

axes[0, 0].hist(df_out4['V17'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 0].axvline(np.median(df_out4['V17']), ls=':', c='g',
label="Median")
axes[0, 0].set_title("V17 Distribution")
axes[0, 1].hist(df_out4['V10'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 1].axvline(np.median(df_out4['V10']), ls=':', c='g',
label="Median")
axes[0, 1].set_title("V10 Distribution")
axes[0, 2].hist(df_out4['V12'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 2].axvline(np.median(df_out4['V12']), ls=':', c='g',
label="Median")
axes[0, 2].set_title("V12 Distribution")
axes[1, 0].hist(df_out4['V16'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 0].set_title("V16 Distribution")
axes[1, 1].hist(df_out4['V14'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 1].set_title("V14 Distribution")
axes[1, 2].hist(df_out4['V3'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 2].set_title("V3 Distribution")
axes[2, 0].hist(df_out4['V7'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 0].set_title("V7 Distribution")
axes[2, 1].hist(df_out4['V11'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 1].set_title("V11 Distribution")
axes[2, 2].hist(df_out4['V4'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 2].set_title("V4 Distribution")
# Add legend for the median lines
axes[0, 0].legend()
# Adjust the layout to avoid overlap
plt.tight_layout()
# Show the plot
plt.show()


from sklearn.ensemble import IsolationForest
df5 = df.copy()
df5 = df5.drop(['Class'], axis=1)


# Import the Isolation Forest model from the scikit-learn library
from sklearn.ensemble import IsolationForest
# Create an Isolation Forest model with specified hyperparameters
# - n_estimators: Number of base estimators in the ensemble (150 in this
case)
# - max_samples: Number of samples to draw from the DataFrame ('auto'
means all samples)
# - contamination: The expected proportion of outliers in the dataset
(0.1 or 10% in this case)
# - max_features: Maximum number of features to consider for each split
(1.0 means all features)
model = IsolationForest(n_estimators=150, max_samples='auto',
contamination=float(0.1), max_features=1.0)
# Fit the Isolation Forest model to the DataFrame 'df5'
model.fit(df5)




# Calculate anomaly scores for each data point in 'df5' using the fitted
Isolation Forest model
scores = model.decision_function(df5)
# Predict whether each data point is an anomaly (outlier) or not
anomaly = model.predict(df5)
# Add the calculated anomaly scores as a new column 'scores' in the 'df5'
DataFrame
df5['scores'] = scores
# Add the binary anomaly predictions as a new column 'anomaly' in the
'df5' DataFrame
df5['anomaly'] = anomaly
# Display the first 10 rows of the updated 'df5' DataFrame, including the
'scores' and 'anomaly' columns
df5.head(10)



# Create a DataFrame 'anomaly' by selecting rows where the 'anomaly'
column is equal to -1 (indicating outliers)
anomaly = df5.loc[df5['anomaly'] == -1]
# Extract the indices of the outlier data points as a list
anomaly_index = list(anomaly.index)
# Print the total number of detected outliers and display it
print('Total number of outliers is:', len(anomaly))


# Select rows from DataFrame 'df5' where the 'anomaly' column is equal to
-1 (indicating outliers)
outliers_df = df5[df5['anomaly'] == -1]
# Display the first 10 rows of the DataFrame containing detected outliers
outliers_df.head(10)


# Create a new DataFrame 'df_out5' by dropping rows with outlier indices
# The 'anomaly_index' list contains the indices of detected outliers
df_out5 = df5.drop(anomaly_index, axis=0).reset_index(drop=True)

# Checking distributions of most important features after dropping
outliers
fig, axes = plt.subplots(nrows=3, ncols=3,figsize=(13,8))
fig.suptitle('Distributions of most important features after dropping
outliers using modified z-score\n', size = 18)
axes[0,0].hist(df_out5['V17'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0,0].axvline(np.median(df_out5['V17']), ls=':', c='g',
label="Median")
axes[0,0].set_title("V17 distribution");
axes[0,1].hist(df_out5['V10'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0,1].axvline(np.median(df_out5['V10']), ls=':', c='g',
label="Median")
axes[0,1].set_title("V10 distribution");
axes[0,2].hist(df_out5['V12'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0,2].axvline(np.median(df_out5['V12']), ls=':', c='g',
label="Median")
axes[0,2].set_title("V12 distribution");
axes[1,0].hist(df_out5['V16'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1,0].set_title("V16 distribution");
axes[1,1].hist(df_out5['V14'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1,1].set_title("V14 distribution");
axes[1,2].hist(df_out5['V3'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1,2].set_title("V3 distribution");
axes[2,0].hist(df_out5['V7'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2,0].set_title("V7 distribution");
axes[2,1].hist(df_out5['V11'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2,1].set_title("V11 distribution");
axes[2,2].hist(df_out5['V4'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2,2].set_title("V4 distribution");
plt.tight_layout()


# Create a copy of the original DataFrame 'df' as 'df6'
df6 = df.copy()
# Drop the 'Class' column from 'df6'
df6 = df6.drop(['Class'], axis=1)
In [39]:
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
# Scale the data using StandardScaler
X = StandardScaler().fit_transform(df6.values)
# Create a DBSCAN clustering model with specified hyperparameters
# 'eps' controls the maximum distance between two samples for one to be
considered as in the neighborhood of the other.
# 'min_samples' sets the minimum number of samples in a neighborhood for
a data point to be considered as a core point.
db = DBSCAN(eps=3.0, min_samples=10).fit(X)
# Extract the cluster labels assigned to each data point
labels = db.labels_

# Calculate the number of clusters in the dataset
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# Print the number of clusters
print('The number of clusters in the dataset is:', n_clusters_)


# Convert the cluster labels to a Pandas Series and count occurrences of
each label
label_counts = pd.Series(labels).value_counts()
# Print the counts of each cluster label
print(label_counts)
Name: count, dtype: int64






%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import numpy as np
from fasteda import fast_eda
from datacleaner import autoclean
import scipy
import scipy.stats as stats
from collections import Counter


df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
Data Inspection

df.head()


df.tail()



df.shape


df.info()


df.describe()


dtype: int64

df.duplicated().sum()



df = df.drop_duplicates()
df.duplicated().sum()



df.drop("Time", axis = 1, inplace = True)
df.head()


feature_list = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
'V10', 'V11',
'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
'V20', 'V21',
'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
'Amount']


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(11, 17))
fig.suptitle('Features vs Class\n', size=18)
# Define the features you want to visualize
features_to_visualize = ['V17', 'V10', 'V12', 'V16', 'V14', 'V3', 'V7',
'V11', 'V4']
# Create boxplots for each feature
for i, feature in enumerate(features_to_visualize):
row, col = i // 3, i % 3 # Calculate the row and column for the
subplot
# Create a boxplot for the feature grouped by 'Class' using the
viridis palette
sns.boxplot(ax=axes[row, col], data=df, x='Class', y=feature,
palette='viridis')
axes[row, col].set_title(f"{feature} Distribution")
# Adjust the layout to avoid overlap
plt.tight_layout()
# Show the plot
plt.show()


def IQR_method(df, n, features):
"""
Identify outliers in a DataFrame using the Tukey IQR method.
Parameters:
df (DataFrame): The input DataFrame.
n (int): The minimum number of outliers in an observation to be
considered.
features (list): List of feature column names to analyze for
outliers.
Returns:
list: A list of indices corresponding to observations with more than
'n' outliers.
"""
outlier_list = []
for column in features:
# 1st quartile (25%)
Q1 = np.percentile(df[column], 25)
# 3rd quartile (75%)
Q3 = np.percentile(df[column], 75)
# Interquartile range (IQR)
IQR = Q3 - Q1
# Outlier step
outlier_step = 1.5 * IQR
# Determine a list of indices of outliers
outlier_list_column = df[(df[column] < Q1 - outlier_step) |
(df[column] > Q3 + outlier_step)].index
# Append the list of outliers
outlier_list.extend(outlier_list_column)
# Count occurrences of each outlier index
outlier_count = Counter(outlier_list)
# Select observations containing more than 'n' outliers
multiple_outliers = [k for k, v in outlier_count.items() if v > n]
# Calculate the total number of outliers
total_outliers = len(multiple_outliers)
print('Total number of outliers is:', total_outliers)
return multiple_outliers

# Detecting outliers using the IQR_method function with a threshold of 1
outlier per observation
Outliers_IQR = IQR_method(df, 1, feature_list)
# Dropping outliers from the DataFrame
df_out = df.drop(Outliers_IQR, axis=0).reset_index(drop=True)


# Set the color palette to 'viridis'
sns.set_palette('viridis')
# Create subplots for visualizing the distributions of important features
after outlier removal
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 8))
fig.suptitle('Distributions of Most Important Features after Dropping
Outliers using IQR Method\n', size=18)
# Plot histograms for each feature
axes[0, 0].hist(df_out['V17'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 0].set_title("V17 Distribution")
axes[0, 1].hist(df_out['V10'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 1].set_title("V10 Distribution")
axes[0, 2].hist(df_out['V12'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 2].set_title("V12 Distribution")
axes[1, 0].hist(df_out['V16'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 0].set_title("V16 Distribution")
axes[1, 1].hist(df_out['V14'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 1].set_title("V14 Distribution")
axes[1, 2].hist(df_out['V3'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 2].set_title("V3 Distribution")
axes[2, 0].hist(df_out['V7'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 0].set_title("V7 Distribution")
axes[2, 1].hist(df_out['V11'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 1].set_title("V11 Distribution")
axes[2, 2].hist(df_out['V4'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 2].set_title("V4 Distribution")
# Adjust the layout to avoid overlap
plt.tight_layout()
# Show the plot
plt.show()

# Set the color palette to 'viridis'
sns.set_palette('viridis')
# Create subplots for visualizing the distributions of important features
after outlier removal
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 8))
fig.suptitle('Distributions of Most Important Features after Dropping
Outliers using IQR Method\n', size=18)
# Define a hue variable (e.g., 'Class') to add color differentiation
hue_variable = 'Class'
# Plot histograms for each feature with hue
for i, feature in enumerate(features_to_visualize):
row, col = i // 3, i % 3 # Calculate the row and column for the
subplot
# Create a histogram for the feature with hue based on 'Class'
sns.histplot(data=df_out, x=feature, bins=60, linewidth=0.5,
edgecolor="white", hue=hue_variable, ax=axes[row, col])
axes[row, col].set_title(f"{feature} Distribution")
# Adjust the layout to avoid overlap
plt.tight_layout()
# Show the plot
plt.show()

def StDev_method(df, n, features):
"""
Identify outliers in a DataFrame using the Standard Deviation method.
Parameters:
df (DataFrame): The input DataFrame.
n (int): The minimum number of outliers in an observation to be
considered.
features (list): List of feature column names to analyze for
outliers.
Returns:
list: A list of indices corresponding to observations with more than
'n' outliers.
"""
outlier_indices = []
for column in features:
# Calculate the mean and standard deviation of the feature column
data_mean = df[column].mean()
data_std = df[column].std()
# Calculate the cutoff value (3 standard deviations from the
mean)
cut_off = data_std * 3
# Determine a list of indices of outliers for the feature column
outlier_list_column = df[(df[column] < data_mean - cut_off) |
(df[column] > data_mean + cut_off)].index
# Append the found outlier indices for the column to the list of
outlier indices
outlier_indices.extend(outlier_list_column)
# Select observations containing more than 'n' outliers
outlier_indices = Counter(outlier_indices)
multiple_outliers = [k for k, v in outlier_indices.items() if v >
n]
# Calculate the total number of outliers
total_outliers = len(multiple_outliers)
print('Total number of outliers is:', total_outliers)
return multiple_outliers

import seaborn as sns
# Calculate the mean and standard deviation of the 'V11' feature
data_mean, data_std = df['V11'].mean(), df['V11'].std()
# Calculate the cutoff value (3 standard deviations from the mean)
cut_off = data_std * 3
# Calculate the lower and upper bounds
lower, upper = data_mean - cut_off, data_mean + cut_off
# Print the lower and upper bound values
print('The lower bound value is:', lower)
print('The upper bound value is:', upper)
# Set the color palette to 'viridis'
sns.set_palette('viridis')
# Create a histogram to visualize the 'V11' feature
plt.figure(figsize=(10, 5))
sns.histplot(x='V11', data=df, bins=70)
# Highlight the regions outside the bounds in red
plt.axvspan(xmin=lower, xmax=df['V11'].min(), alpha=0.2, color='red')
plt.axvspan(xmin=upper, xmax=df['V11'].max(), alpha=0.2, color='red')
# Show the plot
plt.show()

# detecting outliers using the StDev_method
Outliers_StDev = StDev_method(df, 1, feature_list)
# dropping outliers
df_out2 = df.drop(Outliers_StDev, axis=0).reset_index(drop=True)
# Set the color palette to 'viridis'
sns.set_palette('viridis')
# Create subplots for visualizing the distributions of important features
after outlier removal
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 8))
fig.suptitle('Distributions of Most Important Features after Dropping
Outliers using Standard Deviation Method\n', size=18)
# Plot histograms for each feature
axes[0, 0].hist(df_out2['V17'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 0].set_title("V17 Distribution")
axes[0, 1].hist(df_out2['V10'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 1].set_title("V10 Distribution")
axes[0, 2].hist(df_out2['V12'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 2].set_title("V12 Distribution")
axes[1, 0].hist(df_out2['V16'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 0].set_title("V16 Distribution")
axes[1, 1].hist(df_out2['V14'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 1].set_title("V14 Distribution")
axes[1, 2].hist(df_out2['V3'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 2].set_title("V3 Distribution")
axes[2, 0].hist(df_out2['V7'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 0].set_title("V7 Distribution")
axes[2, 1].hist(df_out2['V11'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 1].set_title("V11 Distribution")
axes[2, 2].hist(df_out2['V4'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 2].set_title("V4 Distribution")
# Adjust the layout to avoid overlap
plt.tight_layout()

plt.show()


def z_score_method(df, n, features):
"""
Identify outliers in a DataFrame using the Z-score method.
Parameters:
df (DataFrame): The input DataFrame.
n (int): The minimum number of outliers in an observation to be
considered.
features (list): List of feature column names to analyze for
outliers.
Returns:
list: A list of indices corresponding to observations with more than
'n' outliers.
"""
outlier_list = []
threshold = 3 

for column in features:

data_mean = df[column].mean()
data_std = df[column].std()

z_score = abs((df[column] - data_mean) / data_std)

outlier_list_column = df[z_score > threshold].index

outlier_list.extend(outlier_list_column)

outlier_list = Counter(outlier_list)
multiple_outliers = [k for k, v in outlier_list.items() if v > n]

df1 = df[df.index.isin(multiple_outliers)]
total_outliers = df1.shape[0]
print('Total number of outliers is:', total_outliers)
return multiple_outliers


Outliers_z_score = z_score_method(df, 1, feature_list)

df_out3 = df.drop(Outliers_z_score, axis=0).reset_index(drop=True)


sns.set_palette('viridis')

after outlier removal
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 8))
fig.suptitle('Distributions of Most Important Features after Dropping
Outliers using Z-score\n', size=18)

axes[0, 0].hist(df_out3['V17'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 0].set_title("V17 Distribution")
axes[0, 1].hist(df_out3['V10'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 1].set_title("V10 Distribution")
axes[0, 2].hist(df_out3['V12'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 2].set_title("V12 Distribution")
axes[1, 0].hist(df_out3['V16'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 0].set_title("V16 Distribution")
axes[1, 1].hist(df_out3['V14'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 1].set_title("V14 Distribution")
axes[1, 2].hist(df_out3['V3'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 2].set_title("V3 Distribution")
axes[2, 0].hist(df_out3['V7'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 0].set_title("V7 Distribution")
axes[2, 1].hist(df_out3['V11'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 1].set_title("V11 Distribution")
axes[2, 2].hist(df_out3['V4'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 2].set_title("V4 Distribution")

plt.tight_layout()

plt.show()


from scipy.stats import median_abs_deviation
def z_scoremod_method(df, n, features):
"""
Identify outliers in a DataFrame using the modified z-score method.
Parameters:
df (DataFrame): The input DataFrame.
n (int): The minimum number of outliers in an observation to be
considered.
features (list): List of feature column names to analyze for
outliers.
Returns:
list: A list of indices corresponding to observations with more than
'n' outliers.
"""
outlier_list = []
threshold = 3
for column in features:
data_mean = df[column].mean()
data_mad = median_abs_deviation(df[column])
mod_z_score = abs(0.6745 * (df[column] - data_mean) / data_mad)


outlier_list.extend(outlier_list_column)

outlier_list = Counter(outlier_list)
multiple_outliers = [k for k, v in outlier_list.items() if v > n]

df1 = df[df.index.isin(multiple_outliers)]
total_outliers = df1.shape[0]
print('Total number of outliers is:', total_outliers)
return multiple_outliers

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 8))
fig.suptitle('Distributions of Most Important Features after Dropping
Outliers using Modified Z-score\n', size=18)

axes[0, 0].hist(df_out4['V17'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 0].axvline(np.median(df_out4['V17']), ls=':', c='g',
label="Median")
axes[0, 0].set_title("V17 Distribution")
axes[0, 1].hist(df_out4['V10'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 1].axvline(np.median(df_out4['V10']), ls=':', c='g',
label="Median")
axes[0, 1].set_title("V10 Distribution")
axes[0, 2].hist(df_out4['V12'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0, 2].axvline(np.median(df_out4['V12']), ls=':', c='g',
label="Median")
axes[0, 2].set_title("V12 Distribution")
axes[1, 0].hist(df_out4['V16'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 0].set_title("V16 Distribution")
axes[1, 1].hist(df_out4['V14'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 1].set_title("V14 Distribution")
axes[1, 2].hist(df_out4['V3'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1, 2].set_title("V3 Distribution")
axes[2, 0].hist(df_out4['V7'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 0].set_title("V7 Distribution")
axes[2, 1].hist(df_out4['V11'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 1].set_title("V11 Distribution")
axes[2, 2].hist(df_out4['V4'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2, 2].set_title("V4 Distribution")

axes[0, 0].legend()

plt.tight_layout()

plt.show()


from sklearn.ensemble import IsolationForest
df5 = df.copy()
df5 = df5.drop(['Class'], axis=1)


model = IsolationForest(n_estimators=150, max_samples='auto',
contamination=float(0.1), max_features=1.0)

scores = model.decision_function(df5)
anomaly = model.predict(df5)

df5['scores'] = scores

df5['anomaly'] = anomaly
df5.head(10)

anomaly = df5.loc[df5['anomaly'] == -1]
outliers_df = df5[df5['anomaly'] == -1]





df_out5 = df5.drop(anomaly_index, axis=0).reset_index(drop=True)

fig, axes = plt.subplots(nrows=3, ncols=3,figsize=(13,8))
fig.suptitle('Distributions of most important features after dropping
outliers using modified z-score\n', size = 18)
axes[0,0].hist(df_out5['V17'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0,0].axvline(np.median(df_out5['V17']), ls=':', c='g',
label="Median")
axes[0,0].set_title("V17 distribution");
axes[0,1].hist(df_out5['V10'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0,1].axvline(np.median(df_out5['V10']), ls=':', c='g',
label="Median")
axes[0,1].set_title("V10 distribution");
axes[0,2].hist(df_out5['V12'], bins=60, linewidth=0.5,
edgecolor="white")
axes[0,2].axvline(np.median(df_out5['V12']), ls=':', c='g',
label="Median")
axes[0,2].set_title("V12 distribution");
axes[1,0].hist(df_out5['V16'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1,0].set_title("V16 distribution");
axes[1,1].hist(df_out5['V14'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1,1].set_title("V14 distribution");
axes[1,2].hist(df_out5['V3'], bins=60, linewidth=0.5,
edgecolor="white")
axes[1,2].set_title("V3 distribution");
axes[2,0].hist(df_out5['V7'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2,0].set_title("V7 distribution");
axes[2,1].hist(df_out5['V11'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2,1].set_title("V11 distribution");
axes[2,2].hist(df_out5['V4'], bins=60, linewidth=0.5,
edgecolor="white")
axes[2,2].set_title("V4 distribution");
plt.tight_layout()
df6 = df.copy()
df6 = df6.drop(['Class'], axis=1)
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(df6.values)
db = DBSCAN(eps=3.0, min_samples=10).fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('The number of clusters in the dataset is:', n_clusters_)
label_counts = pd.Series(labels).value_counts()
print(label_counts)
