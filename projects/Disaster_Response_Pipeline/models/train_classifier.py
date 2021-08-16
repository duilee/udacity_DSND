import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
#from xgboost import XGBClassifier
import pickle


def load_data(database_filepath):
    """
    Load dataset from database with read_sql_table
    Define feature and target variables X and Y
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.iloc[:, 4:39]
    
    Y['related'].replace(2, 1, inplace=True)
    
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    """
    Replace white spaces and lemmatize words
   
    """
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    lemmatize = WordNetLemmatizer()
    
    processed_tokens = []
    for tok in tokens:
        processed_token = lemmatize.lemmatize(tok).lower().strip()
        processed_tokens.append(processed_token)
    
    return processed_tokens


def build_model():
    """
    build model and use grid search to find better parameters.
    """
    pipeline = Pipeline([
    
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    
    ])
    
    parameters = {
    'vect__max_df':(0.25, 0.5),
    'vect__ngram_range':((1,1), (1,2)),
    #'tfidf__use_idf': (True)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    predict X_test and
    prints dataframe for model evaluation
    """
    y_pred = model.predict(X_test)
    df2 = pd.DataFrame(y_pred, columns=category_names)
    
    results = pd.DataFrame()
    for column in Y_test.columns:
        lst = []
        lst.append(precision_score(Y_test[column], df2[column]))
        lst.append(recall_score(Y_test[column], df2[column]))
        lst.append(f1_score(Y_test[column], df2[column]))
        results[column] = lst
    
    print(results)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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