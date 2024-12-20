import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data from pickle file
data_dict = pickle.load(open('data.pickle', 'rb'))

# Check the shape of each item in data
print([np.shape(item) for item in data_dict['data']])

# Determine the expected length (most common length)
expected_length = 42  # Change this if the expected length is different

# Fix the shape of each item by either padding or truncating to match the expected length
def fix_data_shape(data, expected_length):
    fixed_data = []
    for item in data:
        # If the item is longer than expected, truncate it
        if len(item) > expected_length:
            fixed_data.append(item[:expected_length])
        # If the item is shorter than expected, pad it with zeros
        elif len(item) < expected_length:
            fixed_data.append(np.pad(item, (0, expected_length - len(item)), mode='constant'))
        else:
            fixed_data.append(item)
    return np.asarray(fixed_data)

# Apply the shape fix to the data
data = fix_data_shape(data_dict['data'], expected_length)

# Convert labels to numpy array
labels = np.asarray(data_dict['labels'])

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the RandomForestClassifier
model = RandomForestClassifier()

model.fit(x_train, y_train)

# Predict on the test set
y_predict = model.predict(x_test)

# Calculate accuracy score
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
    
