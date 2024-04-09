import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import MultiLabelBinarizer


df = pd.read_csv('final_labeled_df.csv')

# Split the data into features (X) and target variable (y)
X = df.drop('LABEL', axis=1)
y = df['LABEL']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Print the decision tree
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print(classification_report(y_test, y_pred))

# Plot the decision tree
plt.figure(figsize=(10, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=y.unique(), rounded=True)
plt.show()

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
plt.title(f'Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Display the value in each cell
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment='center',color='black')
plt.show()



# Train three different decision trees with different max_depth values
clf1 = DecisionTreeClassifier(max_depth=2, random_state=42)
clf1.fit(X_train, y_train)

# Print the decision tree
tree_rules = export_text(clf1, feature_names=list(X.columns))
print(tree_rules)

# Predict the labels for the test set
y_pred1 = clf1.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred1)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print(classification_report(y_test, y_pred1))

# Plot the three decision trees
plt.figure(figsize=(20, 10))
# plt.subplot(1, 3, 1)
plot_tree(clf1, filled=True, feature_names=X.columns, class_names=y.unique(), rounded=True)
plt.title('Decision Tree with max_depth=2')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
plt.title(f'Confusion Matrix max_depth=2')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Display the value in each cell
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment='center',color='black')
plt.show()



clf2 = DecisionTreeClassifier(max_depth=5, random_state=42)
clf2.fit(X_train, y_train)

# Print the decision tree
tree_rules = export_text(clf2, feature_names=list(X.columns))
print(tree_rules)

# Predict the labels for the test set
y_pred2 = clf2.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred2)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print(classification_report(y_test, y_pred2))

# plot
plt.figure(figsize=(40, 10))
plot_tree(clf2, filled=True, feature_names=X.columns, class_names=y.unique(), rounded=True)
plt.title('Decision Tree with max_depth=5')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
plt.title(f'Confusion Matrix for max_depth=5')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Display the value in each cell
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment='center',color='black')
plt.show()


clf3 = DecisionTreeClassifier(max_depth=10, random_state=42)
clf3.fit(X_train, y_train)

# Print the decision tree
tree_rules = export_text(clf3, feature_names=list(X.columns))
print(tree_rules)

# Predict the labels for the test set
y_pred3 = clf1.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred3)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print(classification_report(y_test, y_pred3))

# plot
plt.figure(figsize=(50, 10))
plot_tree(clf3, filled=True, feature_names=X.columns, class_names=y.unique(), rounded=True)
plt.title('Decision Tree with max_depth=10')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
plt.title(f'Confusion Matrix for max_depth=10')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Display the value in each cell
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment='center',color='black')
plt.show()

from sklearn.multioutput import MultiOutputClassifier
# Convert the target variable y into a format suitable for multi-label classification
mlb = MultiLabelBinarizer()
y_binarized = mlb.fit_transform([[label] for label in y])

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.2, random_state=42)

# Initialize and train the multi-output decision tree classifier
multi_target_clf = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))
multi_target_clf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = multi_target_clf.predict(X_test)

# Inverse transform the predicted labels to get the original label format
y_pred_original = mlb.inverse_transform(y_pred)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print(classification_report(y_test, y_pred))

import os

# Create a directory to save the plots
os.makedirs("decision_trees", exist_ok=True)

# Plot and save the decision trees for each output with a different figure size
for i, estimator in enumerate(multi_target_clf.estimators_):
    plt.figure(figsize=(40, 10))  # Adjust the figsize as needed
    plot_tree(estimator, filled=True, feature_names=X.columns, class_names=['Not ' + mlb.classes_[i], mlb.classes_[i]], rounded=True, fontsize=10)
    plt.title(f'Decision Tree for Label: {mlb.classes_[i]}')
    plt.savefig(f"decision_trees/decision_tree_{mlb.classes_[i]}.png")
    plt.close()

