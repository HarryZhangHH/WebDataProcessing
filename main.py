import pandas as pd
import numpy as np
# import texthero as hero
# import nltk
from matplotlib import pyplot as plt
# from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

"""true_news_data = pd.read_csv('/Users/eliasli/Downloads/True.csv')
true_news_data['IS_FAKE'] = 0  # Creating default column
true_news_data.head()

fake_news_data = pd.read_csv('/Users/eliasli/Downloads/Fake.csv')
fake_news_data['IS_FAKE'] = 1 # Creating default column
fake_news_data.head()

news_data = pd.concat([true_news_data,fake_news_data]) # Concating the dataframes

news_data = news_data.sample(frac=1.0).reset_index(drop=True) # Shuffling the data

print(news_data.subject.groupby(news_data['IS_FAKE']).value_counts())

news_data['text'] = hero.lowercase(news_data['text'])

print(hero.top_words(news_data['text'])[:10])

news_data['text'] = hero.remove_punctuation(news_data['text'])

news_data['text'] = hero.remove_whitespace(news_data['text'])

news_data['text'] = news_data['text'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2 and
                                                                word.isalpha() and word != 'reuters']))

# print(news_data['text'])

nltk.download('wordnet')
news_data['text'] = news_data['text'].apply(lambda x: ' '.join([WordNetLemmatizer()
                                                               .lemmatize(word) for word in x.split()]))

print(news_data['text'])

Tfidf_vect = TfidfVectorizer()
Train_content_tfidf = Tfidf_vect.fit_transform(news_data['text'])

SKF = StratifiedKFold(n_splits=5, shuffle=True)

X_train, X_test, Y_train, Y_test = train_test_split(Train_content_tfidf, news_data['IS_FAKE'], test_size=0.2)

LR = LogisticRegression()
LR.fit(X_train, Y_train)
Y_pred = LR.predict(X_test)

print(f' Accuracy Score : {round(metrics.accuracy_score(Y_test,Y_pred)*100,2)}%')
print(metrics.classification_report(Y_test, Y_pred))

dataframe = pd.DataFrame(news_data).astype(str)
dataframe.to_csv("/Users/eliasli/Downloads/News_data.csv", index=False, sep=',')"""

# LSTM Model
news_data = pd.read_csv('./News_data.csv')
print(news_data['text'])
MAX_NB_WORDS = 10000

news_data['text'] = news_data['text'].astype(str)

tokenizer = keras.preprocessing.text.Tokenizer(MAX_NB_WORDS) # Selecting top 10000 words
tokenizer.fit_on_texts(news_data['text'])
train_data = tokenizer.texts_to_sequences(news_data['text'])

word_index = tokenizer.word_index

MAX_SEQUENCE_LENGTH = 600
EMBEDDING_DIM = 30

train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=MAX_SEQUENCE_LENGTH)
# Padding to make equal length of 600 sequences

X_train, X_test, Y_train, Y_test = train_test_split(train_data, news_data['IS_FAKE'], test_size=0.2)

model = keras.Sequential()
model.add(keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))

# Below layers are not required.

# model.add(keras.layers.Conv1D(100, 4, activation='relu'))
# model.add(keras.layers.MaxPooling1D(4))

model.add(keras.layers.LSTM(units=128))

model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, epochs=5, batch_size=1000, validation_split=0.1, verbose=1)
print(model.evaluate(X_test,Y_test, verbose=1))


#saving the model to local
checkpoint_path = "model/model1.ckpt"

y_predicted = model.predict(X_test)
y_predicted_labels = [np.round(i) for i in y_predicted]

sns.heatmap(tf.math.confusion_matrix(labels=Y_test,predictions=y_predicted_labels), annot=True, fmt='d',cmap='YlGnBu')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()



"""if __name__ == '__main__':
    print_hi()
    print("ok")"""