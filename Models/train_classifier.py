import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle
from sklearn.externals import joblib 

def load_data(database_filepath):
    '''
    Input:
    - the file path of the database
    
    Output:
    - a dataframe contains the messages to use as features for the machine learning model
    - a dataframe contains the categories that represent the labels
    - a list of the names of the 36 categories
    '''
    
    engine = create_engine('sqlite:///'+ database_filepath) # ../data/cleaned_dataset.db'
    df = pd.read_sql_table("cleaned_dataset", engine)
    X = df['message'] 
    Y = df.iloc[:,4:]
    category_names=list(Y.columns)
    return X, Y, category_names

def tokenize(text):
    '''
    Input:
    - the raw messages as a dataframe
    
    Output: 
    - a dataframe contains the messages after cleaning process
    '''
    
        text= text.lower()
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)     
        words = word_tokenize(text)
        words = [word for word in words if word not in stopwords.words("english")]
        words = [PorterStemmer().stem(word) for word in words]
        words = [WordNetLemmatizer().lemmatize(word) for word in words]
        words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]
        
        return words

def build_model():
    '''
    Output:
    - a trained model of grid search
    '''
      
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
     
    parameters = {'clf__estimator__n_estimators': [10, 20]}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    #cv.fit(X_train, Y_train)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Input:
    - the trained model
    - the features of the test set
    - the labels of the test set
    - the list of the labels (the 36 categories)
    Output:
    - A classification report
    
    '''    
    y_pred = model.predict(X_test)

    for i in range(len(y_test.columns)):
        print(classification_report(y_test.iloc[:,i].values, y_pred[:,i], target_names=list(y_test.columns)))  
        print(accuracy_score(y_test_T.iloc[:,i].values, y_pred_T[:,i]))

def save_model(model, model_filepath):
    '''
    Input:
    - the trained model
    - the file path of the model
    
    Output:
    - the trained model as pkl file
    '''
    joblib.dump(model, 'models/model.pkl')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        X_train=X_train.transpose()
        
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