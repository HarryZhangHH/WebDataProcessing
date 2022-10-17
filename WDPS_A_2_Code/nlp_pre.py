import pandas as pd
import numpy as np
import os
# import texthero as hero
# import nltk
from matplotlib import pyplot as plt
# from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn import svm

# LSTM Model
news_data = pd.read_csv('./News_data_final.csv')
print(news_data['text'])
print(news_data['title'])
MAX_NB_WORDS = 10000

news_data['text'] = news_data['text'].astype(str)
news_data['title'] = news_data['title'].astype(str)


tokenizer = keras.preprocessing.text.Tokenizer(MAX_NB_WORDS)

tokenizer.fit_on_texts(news_data['text'])
train_data = tokenizer.texts_to_sequences(news_data['text'])
tokenizer2 = keras.preprocessing.text.Tokenizer(MAX_NB_WORDS)

tokenizer2.fit_on_texts(news_data['title'])
train_data2 = tokenizer2.texts_to_sequences(news_data['title'])

word_index = tokenizer.word_index

MAX_SEQUENCE_LENGTH = 600
EMBEDDING_DIM = 30

train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=MAX_SEQUENCE_LENGTH)
train_data2 = keras.preprocessing.sequence.pad_sequences(train_data2, maxlen=20)
# Padding to make equal length of 600 sequences

# print(train_data)
# print(train_data2)

"""new_data = [[]*44898 for _ in range(44898)]
for i in range(44898):
    new_data[i].append(train_data[i])
    new_data[i].append(train_data2[i])"""

# print(new_data[0][0])

text_train, text_test, Lb_train, Lb_test = train_test_split(train_data, news_data['IS_FAKE'], test_size=0.2, random_state=2, shuffle=False)
title_train, title_test, Lt_train, Lt_test = train_test_split(train_data2, news_data['IS_FAKE'], test_size=0.2, random_state=2, shuffle=False)
# print(title_train)
# print(title_test)


def split_text(original):
    text = []
    title = []
    for j in range(len(original)):
        text.append(original[j][0].tolist())
        title.append(original[j][1].tolist())
    return text, title


# print(X_train)

"""text_train, title_train = split_text(X_train)
text_test, title_test = split_text(X_test)

print(type(text_train))
print(type(X_train))

print(type(text_train[0]))"""


def creat_modle():
    model = keras.Sequential()

    model.add(keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))

    model.add(keras.layers.LSTM(units=128))

    model.add(keras.layers.Dense(8, activation='sigmoid'))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def creat_modle_title():
    model = keras.Sequential()

    model.add(keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=20))

    model.add(keras.layers.LSTM(units=8))

    model.add(keras.layers.Dense(4, activation='sigmoid'))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Create a basic model instance - when first train model
model = creat_modle()

# Save the weights
model.summary()

model.fit(text_train, Lb_train, epochs=5, batch_size=1000, validation_split=0.1, verbose=1)

model.save_weights('./model_4_text/my_checkpoint')

print(model.evaluate(text_test, Lb_test, verbose=1))


"""
model = creat_modle_title()
# Save the weights
model.summary()

model.fit(title_train, Lt_train, epochs=3, batch_size=1000, validation_split=0.1, verbose=1)

model.save_weights('./model_4_title/my_checkpoint')

print(model.evaluate(title_test, Lt_test, verbose=1))
"""

# Create a new model instance
# model = creat_modle()
# Display the model's architecture
# model.summary()
# Load the previously saved weights
# model.load_weights('./model/my_checkpoint1')


# Restore the weights
#model.load_weights('./model/my_checkpoint')

"""
print(model.evaluate(X_test,Y_test, verbose=1))

y_predicted = model.predict(X_test)
y_predicted_labels = [np.round(i) for i in y_predicted]
"""

"""
CLF = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
CLF.fit(out_train, y_train)
# predict_results = clf.predict(out_test)
# print(accuracy_score(predict_results, y.astype('int')))
"""


"""sns.heatmap(tf.math.confusion_matrix(labels=Y_test,predictions=y_predicted_labels), annot=True, fmt='d',cmap='YlGnBu')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()"""
