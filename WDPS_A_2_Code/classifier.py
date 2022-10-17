import pandas as pd
import numpy as np
import os
# import texthero as hero
# import nltk
from matplotlib import pyplot as plt
# from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from sklearn import svm
from keras import Model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

# Prepare the input data as vector

text_train, text_test, Lb_train, Lb_test = train_test_split(train_data, news_data['IS_FAKE'], test_size=0.2, random_state=2, shuffle=False)
title_train, title_test, Lt_train, Lt_test = train_test_split(train_data2, news_data['IS_FAKE'], test_size=0.2, random_state=2, shuffle=False)
print(text_train[0])
final_train = np.load('train_dim_12_classifier.npy')
final_test = np.load('test_dim_12_classifier.npy')


LR = LogisticRegression()
LR.fit(final_train, Lb_train)
predict_results = LR.predict(final_test)
print(accuracy_score(predict_results, Lt_test))

[[TN, FP], [FN, TP]] = confusion_matrix(y_true=Lt_test, y_pred=predict_results,
                                            labels=[0, 1]).astype(float)
# print(y.astype('int'))
# print(predict_results)

accuracy = (TP + TN) / (TP + TN + FP + FN)
specificity = TN / (FP + TN)
precision = TP / (TP + FP)
sensivity = TP / (TP + FN)
f1score = 2 * TP / (2 * TP + FP + FN)

print(accuracy)
print(specificity)
print(precision)
print(sensivity)
print(f1score)


y_score = LR.fit(final_train, Lb_train).decision_function(final_test)
# print(y.astype('int'))
# print(predict_results)
print(Lt_test.tolist())
print(type(Lt_test))
print(len(Lt_test))
r = Lt_test.tolist()
r = np.array(r).reshape((8980, 1))
re_1 = [n for a in r for n in a]
# print(re_1)
# for i, k in enumerate(re_1):
#     re_1[i] = k + 1
# print(re_1)
fpr, tpr, threshold = roc_curve(re_1, y_score)
roc_auc = auc(fpr, tpr)
# print(roc_auc)
plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
print(roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.show()
plt.savefig("ROC Curve.png")
# plt.close()

sns.heatmap(tf.math.confusion_matrix(labels=Lt_test.tolist(), predictions=predict_results), annot=True, fmt='d', cmap='YlGnBu')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion matrix On Test Data')
plt.show()

# np.save('label_4_train.npy', Lt_train.tolist())
# np.save('label_4_test.npy', Lt_test.tolist())

inn1 = np.load('label_4_train.npy')
print(inn1)


# ensemble learning

"""clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
# clf.fit(final_train, Lb_train)
    # predict_results = clf.predict(out_test)
    # print(accuracy_score(predict_results, y.astype('int')))

    # conf_mat = confusion_matrix(y.astype('int'), predict_results)

    # print(classification_report(y.astype('int'), predict_results))

dtree = DecisionTreeClassifier(criterion='entropy', splitter='best')
    # dtree.fit(out_train, y_train.astype('int'))

LR = LogisticRegression()
# LR.fit(final_train, Lb_train)

knn = KNeighborsClassifier(n_neighbors=2, weights='uniform', leaf_size=30)
    # knn.fit(out_train, y_train.astype('int'))

RFC = RandomForestClassifier(n_estimators=8)
    # RFC.fit(out_train, y_train.astype('int'))

GU = GaussianNB()
    # GU.fit(out_train, y_train.astype('int'))"""
estimators = [
    ('svm', svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')),
    ('dtree', DecisionTreeClassifier(criterion='entropy', splitter='best')),
    ('lr', LogisticRegression())
]

stacking = StackingClassifier(estimators=estimators, final_estimator=DecisionTreeClassifier(criterion='entropy', splitter='best'))
stacking.fit(final_train, Lb_train)

predict_results = stacking.predict(final_test)
print(accuracy_score(predict_results, Lt_test))

[[TN, FP], [FN, TP]] = confusion_matrix(y_true=Lt_test, y_pred=predict_results,
                                            labels=[0, 1]).astype(float)


accuracy = (TP + TN) / (TP + TN + FP + FN)
specificity = TN / (FP + TN)
precision = TP / (TP + FP)
sensivity = TP / (TP + FN)
f1score = 2 * TP / (2 * TP + FP + FN)

print(accuracy)
print(specificity)
print(precision)
print(sensivity)
print(f1score)
"""
# xgboost
    xg = XGBClassifier(learning_rate=0.6,
                       n_estimators=15,
                       max_depth=4,
                       min_child_weight=6,
                       gamma=0,
                       subsample=0.8,
                       colsample_bytree=0.8,
                       objective='binary:logistic',
                       nthread=4,
                       scale_pos_weight=1,
                       seed=27)
    xg.fit(out_train, y_train)

    y_pred = xg.predict(out_test)
    # predictions = [round(value) for value in y_pred]
    # acc = accuracy_score(y, predictions)
    # f1 = f1_score(y, predictions)
    # pre = precision_score(y, predictions)
    # recall = recall_score(y, predictions)
    # print("acc-xgboost: %.2f%%" % (acc * 100.0))
    predict_results = xg.predict(out_test)
    # print(predict_results)
    # print(classification_report(predict_results, y.astype('int')))
    y = y.reshape((len(y), 1))
    
    y = np.array(y)
    # print(y.astype('int'))
    # print(predict_results)
    # for i, k in enumerate(predict_results):
    #     predict_results[i] = k - 1
    # print(predict_results)
    [[TN, FP], [FN, TP]] = confusion_matrix(y_true=y, y_pred=predict_results,
                                            labels=[0, 1]).astype(float)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    sensivity = TP / (TP + FN)
    f1score = 2 * TP / (2 * TP + FP + FN)
    """


"""
y_train = y_train.reshape((len(y_train), 1))
    y = y.reshape((len(y), 1))
    y = np.array(y)
    y_train = np.array(y_train)
    
    clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf.fit(out_train, y_train)
    # predict_results = clf.predict(out_test)
    # print(accuracy_score(predict_results, y.astype('int')))

    # conf_mat = confusion_matrix(y.astype('int'), predict_results)

    # print(classification_report(y.astype('int'), predict_results))

    dtree = DecisionTreeClassifier(criterion='entropy', splitter='best')
    # dtree.fit(out_train, y_train.astype('int'))

    LR = LogisticRegression()
    LR.fit(out_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=2, weights='uniform', leaf_size=30)
    # knn.fit(out_train, y_train.astype('int'))

    RFC = RandomForestClassifier(n_estimators=8)
    # RFC.fit(out_train, y_train.astype('int'))

    GU = GaussianNB()
    # GU.fit(out_train, y_train.astype('int'))

    stacking = StackingClassifier(classifiers=[LR, clf, dtree], meta_classifier=dtree)

    # stacking = StackingClassifier(classifiers=[knn, LR, dtree], use_probas=True,
    #                               average_probas=False, meta_classifier=GU)
    stacking.fit(out_train, y_train)
    # print('overall:', stacking.score(out_test, y.astype('int')))

    # print('svm:', clf.score(out_test, y.astype('int')))
    # print('knn:', knn.score(out_test, y.astype('int')))
    # print('dtree:', dtree.score(out_test, y.astype('int')))
    # print('LR:', LR.score(out_test, y.astype('int')))
    # print('RFC:', RFC.score(out_test, y.astype('int')))
    # print('GU:', GU.score(out_test, y.astype('int')))
    predict_results = clf.predict(out_test)
    # print(predict_results)
    # print(classification_report(predict_results, y.astype('int')))
    y_1 = y.astype('int')
    # print(y_1)
    # print(predict_results)
    [[TN, FP], [FN, TP]] = confusion_matrix(y_true=y, y_pred=predict_results,
                                            labels=[0, 1]).astype(float)
    # print(y.astype('int'))
    print(predict_results)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    sensivity = TP / (TP + FN)
    f1score = 2 * TP / (2 * TP + FP + FN)
"""


