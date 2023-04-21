# Import necessary modules
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Load the data from the npz files
data = np.load('mfcc_data_40_augmented.npz')
X_train = data['X_train']
y_train = data['y_train']

X_val = data['X_val']
y_val = data['y_val']

# Print the shape of the data
print(X_train.shape)
print(y_train.shape)

image = True

if image:
    # Remove channel dimension, the first dimension, from each image so that shape is 2d
    X_train = [x.squeeze() for x in X_train]
    X_val = [x.squeeze() for x in X_val]

if image:
    print("New shape for each image", X_train[0].shape)
    # Flatten each 2D mfcc into a 1D vector
    X_train = [x.flatten() for x in X_train]
    X_val = [x.flatten() for x in X_val]

    # Convert data back to a numpy array
    X_train = np.array(X_train)
    X_val = np.array(X_val)




# Split the dataset into training, and validation sets
# X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.1, random_state=42)

# Print the shape of the training and validation sets
print(X_train.shape)
print(X_val.shape)

# Print the shape of the training and validation sets
print(y_train.shape)
print(y_val.shape)

# Create a logistic regression model and find the best parameters
parameters = {'C':[0.001, .01, .1, 1, 10, 20, 50]}
model = LogisticRegression(max_iter=50000)
cv = GridSearchCV(model, parameters, verbose=4)
cv.fit(X_train, y_train)
cv.best_params_

# Print the model's accuracy
best_model = cv.best_estimator_
print("Accuracy: ", best_model.score(X_val, y_val))

# Plot the model's coefficients
plt.plot(best_model.coef_.T, 'o')

# Add labels and show the plot
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient magnitude')
plt.show()
