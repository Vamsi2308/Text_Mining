
# #### This work is based on the 'NewHeadlineLIST' which is a list of ll the headlines in  the dataset


import pandas as pd
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from sklearn.decomposition import LatentDirichletAllocation 
import matplotlib.pyplot as plt


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


data = {'text': HeadlineLIST}
df = pd.DataFrame(data)

# Tokenize the text
tokenized_text = [doc.split() for doc in df['text']]

# Create a dictionary and a corpus
dictionary = Dictionary(tokenized_text)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_text]
# print(corpus)

# Perform LDA for k=2, k=3, and k=4
for k in [2, 3, 4, 5, 6, 7]:
    lda_model = LdaModel(corpus, num_topics=k, id2word=dictionary)
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, f'lda_k{k}.html')

# Display the visualizations
for k in [2, 3, 4, 5, 6, 7]:
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.display(vis_data)


##############################################
##
##   LDA Topics Modeling  - Dr. Ami Gates
##
##
#########################################################
filename_1="df_lemmatized.csv"
## Read to DF
df_lemmatized = pd.read_csv(filename_1)
topics=["US Elections", "Trump", "Biden", "Democracy","Republic", "Politics", "United States"]

NumTopics=len(topics)
NUM_TOPICS=NumTopics
lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10000, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
   
lda_Z_DF = lda_model.fit_transform(df_lemmatized)
print(lda_Z_DF.shape)  # (NO_DOCUMENTS, NO_TOPICS)

def print_topics(model, vectorizer, top_n=15):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names_out()[i], topic[i])
                    for i in topic.argsort()[:-top_n - 1:-1]])

print("LDA Model:")
print_topics(lda_model, my_count_vectorizer_lemmatized)


################ Another fun vis for LDA
import numpy as np

filename_2="ColumnNames_lemmatized_df.csv"
## Read to DF
ColumnNames_lemmatized_df = pd.read_csv(filename_2)
# Convert the DataFrame to a NumPy array
ColumnNames_lemmatized = ColumnNames_lemmatized_df.values

word_topic = np.array(lda_model.components_)
#print(word_topic)
word_topic = word_topic.transpose()

num_top_words = 15
vocab_array = np.asarray(ColumnNames_lemmatized)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 7
topics=["US Elections", "Trump", "Biden", "Democracy", "Republic", "Politics", "United States"]

for t in range(NUM_TOPICS):
    plt.subplot(1, NUM_TOPICS, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([])  # remove y-axis markings ('ticks')
    plt.title('{}'.format(topics[t]), fontsize = 10)  # Add topic name to the title
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()

