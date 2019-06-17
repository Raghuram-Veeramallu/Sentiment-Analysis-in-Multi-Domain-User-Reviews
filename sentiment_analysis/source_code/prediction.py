import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re

from sklearn.model_selection import train_test_split

# Library to convert the text into a matrix of TF-IDF and token count.
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Library to apply a pipeline of functions
from sklearn.pipeline import Pipeline

# Libraries to perform Learning on the data
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Library to evaluate the model predictions
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix


categories = ['books', 'dvd', 'electronics', 'kitchen_housewares']

rep_sym = re.compile('[/(){}\[\]\|@,;]')
rep_unw_sym = re.compile('[^0-9a-z #+_]')
stop_words = set(stopwords.words('english'))

def getTrainData(category):
    train_data = pd.read_csv('../data/full_data/'+category+'.csv')
    train_data = train_data.set_index('uniq_id')
    train_data = train_data.drop('Unnamed: 0', axis = 1)
    return train_data

def getTestData(category):
    test_data = pd.read_csv('../data/test_data/test_'+category+'.csv')
    test_data = test_data.set_index('uniq_id')
    test_data = test_data.drop('Unnamed: 0', axis = 1)
    return test_data

def clean_text_process(text):
    # Change all text to lower to avoid ambiguity
    text = text.lower()
    
    # Replace all unwanted symbols by space in text
    text = rep_sym.sub(' ', text) 
    text = rep_unw_sym.sub('', text)
    
    # Remove stopwords from the sentences
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


def clean_data(train_data):
    train_data['summary'] = train_data['summary'].str.replace("[^a-zA-Z#]", " ")
    train_data['review_text'] = train_data['review_text'].str.replace("[^a-zA-Z#]", " ")
    train_data['summary'] = train_data['summary'].apply(clean_text_process)
    train_data['review_text'] = train_data['review_text'].apply(clean_text_process)

    return train_data



def split_data(train_data, output):
    X_train, X_test, y_train, y_test = train_test_split(train_data.review_text,
                                                        output,
                                                        test_size=0.25,
                                                        random_state = 42)

    return (X_train, X_test, y_train, y_test)


def train_sgd_model(X_train, y_train):
    sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None))])
    sgd.fit(X_train, y_train)
    return sgd

def sentiment_process_training(test_cat):
    data_train = pd.DataFrame()
    for cat in categories:
        train_data = getTrainData(cat)
        train_data = clean_data(train_data)
        data_train = data_train.append(train_data)
    X_train, X_test, y_train, y_test = split_data(data_train, data_train.sentiment)
    sgd = train_sgd_model(X_train, y_train)
    
    test_data = getTestData(test_cat)
    test_data = clean_data(test_data)
    y_prediction = sgd.predict(test_data.review_text)

    return (test_data, y_prediction)

def rating_process_training(test_cat):
    data_train = pd.DataFrame()
    for cat in categories:
        train_data = getTrainData(cat)
        train_data = clean_data(train_data)
        data_train = data_train.append(train_data)
    X_train, X_test, y_train, y_test = split_data(data_train, data_train.rating)
    sgd = train_sgd_model(X_train, y_train)
    
    test_data = getTestData(test_cat)
    test_data = clean_data(test_data)
    y_prediction = sgd.predict(test_data.review_text)

    return (test_data, y_prediction)

def write_csv(y_pred_sen, y_pred_rat, test_data, cat):
    sol_data = pd.DataFrame({'uniq_id':test_data.index, 'rating': y_pred_rat, 'sentiment':y_pred_sen})
    sol_data.to_csv(r'data/predictions/'+cat+'.csv')
    return

if __name__ == "__main__":
    category = 'dvd'
    test_data, y_pred_sen = sentiment_process_training(category)
    test_data, y_pred_rat = rating_process_training(category)
    write_csv(y_pred_sen, y_pred_rat, test_data, category)