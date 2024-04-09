
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('final_labeled_df.csv')
# # Assuming df is your DataFrame containing the dataset
X = df.drop(columns=['LABEL'])  # Features
y = df['LABEL']  # Target variable

# # # Filter rows based on two labels
# filtered_df = df[df['LABEL'].isin(['Trump', 'Biden'])]

# # #Print the filtered dataframe
# print(filtered_df)
# # Assuming df is your DataFrame containing the dataset
# X = filtered_df.drop(columns=['LABEL'])  # Features
# y = filtered_df['LABEL']  # Target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import matplotlib.pyplot as plt
# Define kernel types and costs
kernels = ['linear', 'poly', 'rbf']
costs = [0.1, 1, 10]

# Iterate over each kernel and cost
for kernel in kernels:
    for cost in costs:
        # Print kernel and cost
        print(f'Kernel: {kernel}, Cost: {cost}')

        # Print training data
        print('Training Data:')
        print(X_train.head())
        print(y_train.head())

        # Train SVM model
        svm_model = SVC(kernel=kernel, C=cost)
        svm_model.fit(X_train, y_train)

        # Print test data
        print('Test Data:')
        print(X_test.head())
        print(y_test.head())

        # Predict using the model
        y_pred = svm_model.predict(X_test)

        # # Print predicted labels
        # print('Predicted Labels:')
        # print(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        # Print accuracy for the specific cost
        print(f'Accuracy for {kernel.capitalize()} Kernel with Cost={cost}: {accuracy}')
        print("")
        
        # Print classification report
        print(f'Classification Report for {kernel.capitalize()} Kernel with Cost={cost}:\n')
        print(classification_report(y_test, y_pred))

        # Print confusion matrix
        print(f'Confusion Matrix for {kernel.capitalize()} Kernel with Cost={cost}:\n')
        cm = confusion_matrix(y_test, y_pred)
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix for {kernel.capitalize()} Kernel with Cost={cost}')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        # plt.xticks([0, 1], ['Negative', 'Positive'])
        # plt.yticks([0, 1], ['Negative', 'Positive'])

        # Display the value in each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment='center',
                         color='black')

        plt.show()
        print(confusion_matrix(y_test, y_pred))
        print("----------------------------------------")
    print("")


#######################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('final_labeled_df.csv')

# Filter rows based on two labels
filtered_df = df[df['LABEL'].isin(['Trump', 'Biden'])]

# Assuming df is your DataFrame containing the dataset
X = filtered_df.drop(columns=['LABEL'])  # Features
y = filtered_df['LABEL']  # Target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Define kernel types and costs
kernels = ['linear', 'poly', 'rbf']
costs = [0.1, 1, 10, 25, 50, 100]

# Train SVM model on reduced 2D data
for kernel in kernels:
    for cost in costs:
        svm_model = SVC(kernel=kernel, C=cost)
        svm_model.fit(X_train_pca, y_train)

        # Predict using the model
        y_pred = svm_model.predict(X_test_pca)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Print accuracy for the specific cost
        print(f'Accuracy for {kernel.capitalize()} Kernel with Cost={cost}: {accuracy}')
        print("")

        # Print classification report
        print(f'Classification Report for {kernel.capitalize()} Kernel with Cost={cost}:\n')
        print(classification_report(y_test, y_pred))

        # Print confusion matrix
        print(f'Confusion Matrix for {kernel.capitalize()} Kernel with Cost={cost}:\n')
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        if kernel == 'linear':
            # Plot decision boundaries for all costs with linear kernel
            plt.figure()
            h = .02  # step size in the mesh
            x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
            y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            coef = svm_model.coef_[0]
            intercept = svm_model.intercept_[0]
            Z = coef[0] * xx + coef[1] * yy + intercept
            Z[Z >= 0] = 1
            Z[Z < 0] = 0
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
            plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())
            plt.title(f'SVM Decision Boundaries for {kernel.capitalize()} Kernel with Cost={cost}')
            plt.show()

    print("")

######################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('final_labeled_df.csv')

# Filter rows based on two labels
filtered_df = df[df['LABEL'].isin(['Trump', 'Biden'])]

# Assuming df is your DataFrame containing the dataset
X = filtered_df.drop(columns=['LABEL'])  # Features
y = filtered_df['LABEL']  # Target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Define kernel types and costs
kernels = ['linear', 'poly', 'rbf']
costs = [0.1, 1, 10, 25, 50, 100]

# Lists to store accuracy values for each cost
linear_accuracies = []
poly_accuracies = []
rbf_accuracies = []

# Train SVM model on reduced 2D data
for kernel in kernels:
    accuracies = []
    for cost in costs:
        svm_model = SVC(kernel=kernel, C=cost)
        svm_model.fit(X_train_pca, y_train)

        # Predict using the model
        y_pred = svm_model.predict(X_test_pca)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    if kernel == 'linear':
        linear_accuracies = accuracies
    elif kernel == 'poly':
        poly_accuracies = accuracies
    elif kernel == 'rbf':
        rbf_accuracies = accuracies

# Plotting the accuracies for different costs
plt.figure(figsize=(10, 6))
plt.plot(costs, linear_accuracies, marker='o', label='Linear Kernel')
plt.plot(costs, poly_accuracies, marker='s', label='Polynomial Kernel')
plt.plot(costs, rbf_accuracies, marker='^', label='RBF Kernel')
plt.xlabel('Cost')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Cost for Different Kernels')
plt.legend()
plt.grid(True)
plt.show()

