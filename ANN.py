# Import the pandas library for reading and manipulating data
import pandas as pd

# Read data from the CSV file while handling missing values
df = pd.read_csv('datak.csv', na_values=["na", "-", "NaN"])

# Display concise information about the read data
df.info()

# Display the first few rows of data
df.head()

# Import data from the CSV file (handling missing values)
df = pd.read_csv('datak.csv')

# Replace missing values with NaN
df = pd.read_csv('datak.csv', na_values=["na", "-", "NaN"])

# Specify the CSV file name to be read
filename = 'datak.csv'

# Read data from the CSV file and replace missing values with NaN
df = pd.read_csv(filename, na_values=["na", "-", "NaN"])

# Display concise information about the read data
df.info()

# Display the first few rows of data
df.head()

# List of features to be used in analysis
features = ['pac', 'sho', 'pas', 'dri', 'def', 'phy']

# Retrieve feature values from data as a numpy array
data_features = df.loc[:, features].values

# Display the dimension of feature data
n = len(data_features)
print("Number of data points:", n)

# Normalize feature data by dividing each value by 100
data_normalized = data_features[:, :] / 100

# Display normalized feature data
print("Normalized Feature Data:")
print(data_normalized)

# Number of dimensions in feature data
n_dimensions = len(data_normalized[0])
print("Number of Feature Dimensions:", n_dimensions)

# Target column name
target_column = 'lbl'

# Retrieve target values from data as a numpy array
data_target = df.loc[:, target_column].values

# Display target values
print("Target Data:")
print(data_target)

# Learning rate value
learning_rate = 0.9

# Initialize initial weights
weights = [1, 1, 1, 1, 1, 1]

# Threshold value
threshold = 0

# List to store errors
error_list = []

# Model training
for i in range(n):
    V = 0
    for k in range(n_dimensions):
        V += data_normalized[i][k] * weights[k]
    if V < threshold:
        prediction = 0
    else:
        prediction = 1
    error = data_target[i] - prediction
    error_list.append(error)
    for k in range(n_dimensions):
        weights[k] += learning_rate * error * data_normalized[i][k]

# Iterative model refinement
previous_errors = []
loop = 0
while error_list != previous_errors:
    previous_errors = error_list.copy()
    error_list = []
    for i in range(n):
        V = 0
        for k in range(n_dimensions):
            V += data_normalized[i][k] * weights[k]
        if V < threshold:
            prediction = 0
        else:
            prediction = 1
        error = data_target[i] - prediction
        error_list.append(error)
        for k in range(n_dimensions):
            weights[k] += learning_rate * error * data_normalized[i][k]
    loop += 1
    print("Loop:", loop)

# Import data for testing
dff = pd.read_csv('datak1.csv', na_values=["na", "-", "NaN"])

# Retrieve feature values from testing data as a numpy array
dff_features = dff.loc[:, features].values

# Normalize testing feature data by dividing each value by 100
dff_normalized = dff_features[:, :] / 100

# Retrieve target values from testing data as a numpy array
dff_target = dff.loc[:, target_column].values

# List to store predictions
predictions = []

# Model testing
for i in range(len(dff)):
    V = 0
    for k in range(n_dimensions):
        V += dff_normalized[i][k] * weights[k]
    if V < threshold:
        prediction = 0
    else:
        prediction = 1
    predictions.append(prediction)

# Calculate accuracy
correct_predictions = sum(1 for i in range(nob) if predictions[i] == dff_target[i])
accuracy = (correct_predictions / nob) * 100
print('Accuracy of the data is:', accuracy, 'percent')

# Predict the position of a new player
input_features = []
for i in range(n_dimensions):
    value = float(input('Enter a value for feature ' + features[i] + ': '))
    input_features.append(value)
V = 0
for k in range(n_dimensions):
    V += input_features[k] * weights[k]
if V < threshold:
    prediction = 0
    print('This player cannot play in the striker position')
else:
    prediction = 1
    print('This player can play in the striker position')
