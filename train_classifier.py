# import packages

import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from joblib import dump
import sqlite3


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# load data

def load_data(database_filepath):
    
    engine = create_engine('sqlite:///'+database_filepath);
    df = pd.read_sql('select * from myNLP',engine).dropna(axis=0);
    X = df['message'];
    Y = df.iloc[:,4:].values;
     
    
    category_names= df.columns[4:]
    
    return X, Y, category_names

# define tokenizer

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    text = (re.sub(r'[^a-zA-Z0-9]',' ',text))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# model building

def build_model():

    model =  Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(XGBClassifier(n_estimators =20,max_depth = 2, min_child_weight = 6)))
    ]);
    
    
    return model 

# model evaluation

def evaluate_model(model, X_test, Y_test, category_names):
    
    pred = model.predict(X_test);
    

    pred=pd.DataFrame(pred,columns=category_names);
    y_test=pd.DataFrame(Y_test,columns=category_names);

    for a in category_names:
        print(a)
        print(classification_report(y_test[a],pred[a]))
   
# saving model as pickle

def save_model(model, model_filepath):
    
    dump(model, model_filepath) 

# main function
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()