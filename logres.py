# Import necessary modules
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Load the data from the npz files
data = np.load('mfcc_data_20.npz')
X = data['X']
y = data['y']

# Print the shape of the data
print(X.shape)
print(y.shape)

image = False

if image:
    # Remove channel dimension, the first dimension, from each image so that shape is 2d
    X = [x.squeeze() for x in X]

if image:
    print("New shape for each image", X[0].shape)

# Convert X back to a numpy array
X = np.array(X)

# Flatten each 2D mel spectrogram into a 1D vector
X_flat = [x.flatten() for x in X]

# Convert X_flat back to a numpy array
X_flat = np.array(X_flat)

# Split the dataset into training, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.1, random_state=42)

# Print the shape of the training and validation sets
print(X_train.shape)
print(X_test.shape)

# Print the shape of the training and validation sets
print(y_train.shape)
print(y_test.shape)

# Create a logistic regression model and find the best parameters
parameters = {'C':[0.001, .01, .1, 1, 10, 20, 50]}
model = LogisticRegression(max_iter=50000)
cv = GridSearchCV(model, parameters, verbose=3)
cv.fit(X_train, y_train)
cv.best_params_

# Print the model's accuracy
best_model = cv.best_estimator_
print("Accuracy: ", best_model.score(X_test, y_test))

# Plot the model's coefficients
plt.plot(best_model.coef_.T, 'o')

# Add labels and show the plot
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient magnitude')
plt.show()
