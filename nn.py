
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GlobalMaxPooling1D
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("final_labeled_df.csv")
data

# Convert 'LABEL' column to one-hot encoding
label_encoder = LabelEncoder()
data['LABEL'] = label_encoder.fit_transform(data['LABEL'])

# Split the data into features (X) and target variable (y)
X = data.drop('LABEL', axis=1)
y = data['LABEL']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert DataFrames to Numpy arrays
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Define the Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_np.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
Hist = model.fit(X_train_np, y_train_np, epochs=5, batch_size=32, validation_data=(X_test_np, y_test_np))

# # Evaluate the Model
# loss, accuracy = model.evaluate(X_test_np, y_test_np)
# print(f'Loss: {loss}, Accuracy: {accuracy}')

plt.plot(Hist.history['accuracy'], label='accuracy')
plt.plot(Hist.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

##Test and Model
Test_Loss, Test_Accuracy = model.evaluate(X_test_np, y_test_np)

## Save the Model
model.save("My_Example_NN_Model")

## Predictions
predictions=model.predict(X_test_np)
## For predictions >=.5 --> 1 else 0
print(predictions)
print(type(predictions))
print(type(y_test_np))
print(y_test_np)
predictions[predictions >= .5] = 1
predictions[predictions < .5] = 0
#predictions = np.array(predictions)
print(predictions)
print(confusion_matrix(predictions, y_test_np))

"""**ANN**"""

# Define and compile an ANN model
model_ann = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')  # Assuming 6 classes
])

model_ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_ann.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss_ann, accuracy_ann = model_ann.evaluate(X_test, y_test)
print(f'ANN Loss: {loss_ann}, Accuracy: {accuracy_ann}')

"""**CNN**"""

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

import numpy as np

# Convert DataFrame columns to NumPy arrays
X_train_array = X_train.to_numpy()
X_test_array = X_test.to_numpy()

# Reshape input data for CNN (assuming 1D CNN)
X_train_cnn = X_train_array.reshape(X_train_array.shape[0], X_train_array.shape[1], 1)
X_test_cnn = X_test_array.reshape(X_test_array.shape[0], X_test_array.shape[1], 1)


# Define and compile a 1D CNN model
model_cnn = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')  # Assuming 6 classes
])

model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_cnn.fit(X_train_cnn, y_train, epochs=5, batch_size=32, validation_data=(X_test_cnn, y_test))

# Evaluate the model
loss_cnn, accuracy_cnn = model_cnn.evaluate(X_test_cnn, y_test)
print(f'CNN Loss: {loss_cnn}, Accuracy: {accuracy_cnn}')

"""**RNN - LSTM**"""

from tensorflow.keras.layers import LSTM

# Reshape input data for RNN (assuming 3D input)
X_train_rnn = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_rnn = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)

# Define and compile an RNN model using LSTM
model_rnn = Sequential([
    LSTM(64, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
    Dense(6, activation='softmax')  # Assuming 6 classes
])

model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_rnn.fit(X_train_rnn, y_train, epochs=5, batch_size=32, validation_data=(X_test_rnn, y_test))

# Evaluate the model
loss_rnn, accuracy_rnn = model_rnn.evaluate(X_test_rnn, y_test)
print(f'RNN Loss: {loss_rnn}, Accuracy: {accuracy_rnn}')

from tensorflow.keras.callbacks import TensorBoard

# For CNN model
tensorboard_callback_cnn = TensorBoard(log_dir='./logs_cnn', histogram_freq=1)

# For RNN model
tensorboard_callback_rnn = TensorBoard(log_dir='./logs_rnn', histogram_freq=1)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs

# For CNN model
model_cnn.fit(X_train_cnn, y_train, epochs=5, batch_size=32,
               validation_data=(X_test_cnn, y_test), callbacks=[tensorboard_callback_cnn])

# For RNN model
model_rnn.fit(X_train_rnn, y_train, epochs=5, batch_size=32,
               validation_data=(X_test_rnn, y_test), callbacks=[tensorboard_callback_rnn])

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data
data = pd.read_csv("final_labeled_df.csv")

# Split the data into features (X) and target variable (y)
X = data.drop('LABEL', axis=1)
y = data['LABEL']

# Convert DataFrame columns to NumPy arrays
X_train_array = X_train.to_numpy()
X_test_array = X_test.to_numpy()

# Convert labels to numerical values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define and compile an ANN model
model_ann = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_array.shape[1],)),
    Dense(6, activation='softmax')  # Assuming 6 classes
])
model_ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define and compile a perceptron model
model_perceptron = Sequential([
    Dense(1, activation='sigmoid', input_shape=(X_train_array.shape[1],))
])
model_perceptron.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define and compile a CNN model
model_cnn = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train_array.shape[1], 1)),
    GlobalMaxPooling1D(),
    Dense(6, activation='softmax')  # Assuming 6 classes
])
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define and compile an RNN model using LSTM
model_rnn = Sequential([
    LSTM(64, input_shape=(X_train_array.shape[1], 1)),
    Dense(6, activation='softmax')  # Assuming 6 classes
])
model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# TensorBoard callbacks
tensorboard_callback_ann = TensorBoard(log_dir='./logs_ann', histogram_freq=1)
tensorboard_callback_perceptron = TensorBoard(log_dir='./logs_perceptron', histogram_freq=1)
tensorboard_callback_cnn = TensorBoard(log_dir='./logs_cnn', histogram_freq=1)
tensorboard_callback_rnn = TensorBoard(log_dir='./logs_rnn', histogram_freq=1)

# Train the models
model_ann.fit(X_train_array, y_train_encoded, epochs=5, batch_size=32, validation_data=(X_test_array, y_test_encoded), callbacks=[tensorboard_callback_ann])
model_perceptron.fit(X_train_array, y_train_encoded, epochs=5, batch_size=32, validation_data=(X_test_array, y_test_encoded), callbacks=[tensorboard_callback_perceptron])
model_cnn.fit(X_train_array, y_train_encoded, epochs=5, batch_size=32, validation_data=(X_test_array, y_test_encoded), callbacks=[tensorboard_callback_cnn])
model_rnn.fit(X_train_array, y_train_encoded, epochs=5, batch_size=32, validation_data=(X_test_array, y_test_encoded), callbacks=[tensorboard_callback_rnn])

# Evaluate the models
loss_ann, accuracy_ann = model_ann.evaluate(X_test_array, y_test_encoded)
loss_perceptron, accuracy_perceptron = model_perceptron.evaluate(X_test_array, y_test_encoded)
loss_cnn, accuracy_cnn = model_cnn.evaluate(X_test_array, y_test_encoded)
loss_rnn, accuracy_rnn = model_rnn.evaluate(X_test_array, y_test_encoded)

print(f'ANN Loss: {loss_ann}, Accuracy: {accuracy_ann}')
print(f'Perceptron Loss: {loss_perceptron}, Accuracy: {accuracy_perceptron}')
print(f'CNN Loss: {loss_cnn}, Accuracy: {accuracy_cnn}')
print(f'RNN Loss: {loss_rnn}, Accuracy: {accuracy_rnn}')



# Commented out IPython magic to ensure Python compatibility.
# %reload_ext tensorboard
# %tensorboard --logdir logs

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder
# from gensim.sklearn_api import LdaTransformer
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

# Load the data
data = pd.read_csv("final_labeled_df.csv")

# Split the data into features (X) and target variable (y)
X = data.drop('LABEL', axis=1)
y = data['LABEL']

# Convert DataFrame columns to NumPy arrays
X_train_array = X_train.to_numpy()
X_test_array = X_test.to_numpy()

# Convert labels to numerical values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# # Topic modeling
# lda_model = LdaTransformer(num_topics=5)
# X_train_lda = lda_model.fit_transform(X_train_array)
# X_test_lda = lda_model.transform(X_test_array)

# Sentiment analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

sentiment_analyzer = SentimentIntensityAnalyzer()
def get_vader_sentiment(text):
    return sentiment_analyzer.polarity_scores(text)['compound']

X_train_sentiment = np.array([get_sentiment(str(text)) for text in X_train_array]).reshape(-1, 1)
X_test_sentiment = np.array([get_sentiment(str(text)) for text in X_test_array]).reshape(-1, 1)

# Define and compile an ANN model
model_ann = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_array.shape[1],)),
    Dense(6, activation='softmax')  # Assuming 6 classes
])
model_ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define and compile a perceptron model
model_perceptron = Sequential([
    Dense(1, activation='sigmoid', input_shape=(X_train_array.shape[1],))
])
model_perceptron.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define and compile a CNN model
model_cnn = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train_array.shape[1], 1)),
    GlobalMaxPooling1D(),
    Dense(6, activation='softmax')  # Assuming 6 classes
])
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define and compile an RNN model using LSTM
model_rnn = Sequential([
    LSTM(64, input_shape=(X_train_array.shape[1], 1)),
    Dense(6, activation='softmax')  # Assuming 6 classes
])
model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# TensorBoard callbacks
tensorboard_callback_ann = TensorBoard(log_dir='./logs_ann', histogram_freq=1)
tensorboard_callback_perceptron = TensorBoard(log_dir='./logs_perceptron', histogram_freq=1)
tensorboard_callback_cnn = TensorBoard(log_dir='./logs_cnn', histogram_freq=1)
tensorboard_callback_rnn = TensorBoard(log_dir='./logs_rnn', histogram_freq=1)

# Train the models
model_ann.fit(X_train_array, y_train_encoded, epochs=5, batch_size=32, validation_data=(X_test_array, y_test_encoded), callbacks=[tensorboard_callback_ann])
model_perceptron.fit(X_train_array, y_train_encoded, epochs=5, batch_size=32, validation_data=(X_test_array, y_test_encoded), callbacks=[tensorboard_callback_perceptron])
model_cnn.fit(X_train_array, y_train_encoded, epochs=5, batch_size=32, validation_data=(X_test_array, y_test_encoded), callbacks=[tensorboard_callback_cnn])
model_rnn.fit(X_train_array, y_train_encoded, epochs=5, batch_size=32, validation_data=(X_test_array, y_test_encoded), callbacks=[tensorboard_callback_rnn])

# Evaluate the models
loss_ann, accuracy_ann = model_ann.evaluate(X_test_array, y_test_encoded)
loss_perceptron, accuracy_perceptron = model_perceptron.evaluate(X_test_array, y_test_encoded)
loss_cnn, accuracy_cnn = model_cnn.evaluate(X_test_array, y_test_encoded)
loss_rnn, accuracy_rnn = model_rnn.evaluate(X_test_array, y_test_encoded)

print(f'ANN Loss: {loss_ann}, Accuracy: {accuracy_ann}')
print(f'Perceptron Loss: {loss_perceptron}, Accuracy: {accuracy_perceptron}')
print(f'CNN Loss: {loss_cnn}, Accuracy: {accuracy_cnn}')
print(f'RNN Loss: {loss_rnn}, Accuracy: {accuracy_rnn}')

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext tensorboard
# %tensorboard --logdir logs

import nltk
nltk.download('vader_lexicon')