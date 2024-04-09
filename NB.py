
# ### Naive Baiyes

# Imporitng In-built packages

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Load the data
# Importing the labeled and the Lemmatized dataframe which is created during the data preparation process.

df = pd.read_csv('final_labeled_df.csv')
print("Original_df_shape:",df.shape)

# Separate features (X) and labels (y)
X = df.drop(columns=["LABEL"])
y = df["LABEL"]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the train and test data
print("Training Data Shape:", X_train.shape)
print("Training Data head:", X_train.head())
print("Test Data Shape:", X_test.shape)
print("Test Data head:", X_test.head())
print("")


# ### Multinomial Naive Bayes model


# Train a Naive Bayes model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Generate a word cloud for the predicted labels
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = None, 
                min_font_size = 10).generate(' '.join(y_pred))

# Display the word cloud
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix Multinomial NB :")
# print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
plt.title(f'Confusion Matrix Multinomial NB :')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Display the value in each cell
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment='center',color='black')
plt.show()


# ### Gaussian Naive Bayes model


# Train a Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict on the test data
y_pred_gnb = gnb.predict(X_test)

# Generate a word cloud for the predicted labels (GaussianNB)
wordcloud_gnb = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(' '.join(y_pred_gnb))

# Display the word cloud (GaussianNB)
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud_gnb)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# Evaluate the models
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print("Accuracy (GaussianNB):", accuracy_gnb)
print("")

# Confusion matrix
conf_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)

# print("Confusion Matrix (GaussianNB):")
# print(conf_matrix_gnb)
# print("")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_gnb, interpolation='nearest', cmap='Blues')
plt.title(f'Confusion Matrix GaussianNB')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Display the value in each cell
for i in range(conf_matrix_gnb.shape[0]):
    for j in range(conf_matrix_gnb.shape[1]):
        plt.text(j, i, format(conf_matrix_gnb[i, j], 'd'), horizontalalignment='center',color='black')
plt.show()


# ### Bernoulli Naive Bayes

# Print the shape of the train and test data
print("Training Data Shape:", X_train.shape)
print("Test Data Shape:", X_test.shape)
print("")

# Train a Bernoulli Naive Bayes model
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

# Predict on the test data
y_pred_bnb = bnb.predict(X_test)

# Generate a word cloud for the predicted labels (BernoulliNB)
wordcloud_bnb = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(' '.join(y_pred_bnb))

# Display the word cloud (BernoulliNB)
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud_bnb)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# Evaluate the models
accuracy_bnb = accuracy_score(y_test, y_pred_bnb)
print("Accuracy (BernoulliNB):", accuracy_bnb)
print("")

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred_bnb)
# print("Confusion Matrix (BernoulliNB):")
# print(cm)

# Plot confusion matrix
# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title(f'Confusion Matrix BernoulliNB')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Display the value in each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center',color='black')
plt.show()

