
# #### This work is based on the 'NewHeadlineLIST' which is a list of ll the headlines in  the dataset

import os
import re   ## for regular expressions
import nltk
import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


## Need to import some code related to cleaning
filename="main_data.csv"
## Read to DF
df = pd.read_csv(filename)

## REMOVE any rows with NaN in them
df = df.dropna()
print(df["Headline"])
print(df.shape)

### Tokenize and Vectorize the Headlines
## Create the list of headlines
## Keep the labels!
HeadlineLIST=[]
LabelLIST=[]

for nexthead, nextlabel in zip(df["Headline"], df["LABEL"]):
    HeadlineLIST.append(nexthead)
    LabelLIST.append(nextlabel)

print("\n The headline list is:\n")
print(HeadlineLIST)

print("\n The label list is:\n")
print(LabelLIST)    

##########################################
## Remove all words that match the topics.
## For example, if the topics are food and covid
## remove these exact words.
##
## We will need to do this by hand. 

topics=["US Elections", "Trump", "Biden", "Democracy","Republic", "Politics", "United States"]
NewHeadlineLIST=[]

for element in HeadlineLIST:
    # print(element)
    # print(type(element))
    ## make into list
    AllWords=element.split(" ")
    # print(AllWords)
    
    ## Now remove words that are in your topics
    NewWordsList=[]
    for word in AllWords:
        # print(word)
        word=word.lower()
        if word in topics:
            print(word)
        else:
            NewWordsList.append(word)
            
    ##turn back to string
    NewWords=" ".join(NewWordsList)
    ## Place into NewHeadlineLIST
    NewHeadlineLIST.append(NewWords)

## Set the HeadlineLIST to the new one
HeadlineLIST=NewHeadlineLIST
print(HeadlineLIST) 



# Sample text data
text_data = HeadlineLIST

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data)


# Calculate SSE for different values of k
sse = []
silhouette_scores = []
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot the SSE curve
plt.figure(figsize=(10, 5))
plt.plot(range(2, 6), sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal k')
plt.xticks(range(2, 6))
plt.show()

# Plot the silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(range(2, 6), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.xticks(range(2, 6))
plt.show()

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(X)

# # Print cluster assignments
# for i, text in enumerate(text_data):
#     print(f"Text: {text}, Cluster: {clusters[i]}")

# Evaluate clustering using silhouette score
silhouette_score = silhouette_score(X, clusters)
print(f"Silhouette Score: {silhouette_score}")


# In[4]:



# Sample text data
text_data = HeadlineLIST

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data).toarray()

# Apply PCA to reduce the dimensionality to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform K-means clustering with the optimal k value
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X)

# Create a DataFrame with the PCA components and cluster assignments
df = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'Cluster': clusters})

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis')
plt.title('Clusters of Text Data')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.colorbar()
plt.show()


# In[13]:


from sklearn import preprocessing
from sklearn.metrics import silhouette_score
## Our DF
filename = "df_lemmatized.csv"
My_Orig_DF = pd.read_csv(filename)
print(My_Orig_DF.head())

#from sklearn.metrics import silhouette_samples, silhouette_score
#from sklearn.cluster import KMeans

My_KMean= KMeans(n_clusters=3)
My_KMean.fit(My_Orig_DF)
My_labels=My_KMean.predict(My_Orig_DF)
# print(My_labels)

#from sklearn import preprocessing
#from sklearn.cluster import KMeans
#import seaborn as sns

My_KMean2 = KMeans(n_clusters=4).fit(preprocessing.normalize(My_Orig_DF))
My_KMean2.fit(My_Orig_DF)
My_labels2=My_KMean2.predict(My_Orig_DF)
# print(My_labels2)

My_KMean3= KMeans(n_clusters=3)
My_KMean3.fit(My_Orig_DF)
My_labels3=My_KMean3.predict(My_Orig_DF)
# Assuming My_Orig_DF is your original dataframe and My_labels3 are the cluster labels
silhouette_avg = silhouette_score(My_Orig_DF, My_labels3)
print("Silhouette Score for k = 3 \n", silhouette_avg)





# Sample text data
text_data = HeadlineLIST

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data).toarray()

# Apply PCA to reduce the dimensionality to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform K-means clustering with the optimal k value
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X)

# Create a DataFrame with the PCA components and cluster assignments
df = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'Cluster': clusters})

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis')
plt.title('Clusters of Text Data')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.colorbar()
plt.show()

# Plot the Lloyd plot
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

plt.figure(figsize=(10,6))
plt.plot(range(1, 11), distortions, marker='o')
plt.title('Lloyd Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


filename = "df_lemmatized.csv"
kmeans_object_Count = sklearn.cluster.KMeans(n_clusters=2)
# print(kmeans_object_Count)
DF_Count = pd.read_csv(filename)
kmeans_object_Count.fit(DF_Count)
# Get cluster assignment labels
labels = kmeans_object_Count.labels_
prediction_kmeans = kmeans_object_Count.predict(DF_Count)
#print(labels)
print(prediction_kmeans)
# Format results as a DataFrame
Myresults = pd.DataFrame([DF_Count.index,labels]).T
print(Myresults)

