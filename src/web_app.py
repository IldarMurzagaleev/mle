from flask import Flask, render_template, request, jsonify
import subprocess
from src import predict
from src import utils


import argparse
import configparser
from datetime import datetime
import os
import json
import pandas as pd
import pickle
import shutil
import sys
import time
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer



def vectorize(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple:
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    y_train['label_id'] = y_train['label'].factorize()[0]
    y_test['label_id'] = y_test['label'].factorize()[0]

    label_id_df = y_train[['label', 'label_id']].drop_duplicates().sort_values('label_id')
    label_to_id = dict(label_id_df.values)
    id_to_label = dict(label_id_df[['label_id', 'label']].values)

    features = tfidf.fit_transform(X_train.text).toarray() # Remaps the words in the train articles in the text column of 
                                                           # data frame into features (superset of words) with an importance assigned 
                                                           # based on each words frequency in the document and across documents
    labels = y_train.label_id                              # represents the category of each of the all train articles
    test_labels = y_test.label_id
    test_features = tfidf.transform(X_test.text.tolist())
    return (tfidf, features, test_features, labels, test_labels, label_to_id, id_to_label)

class WebPredictor():

    def __init__(self) -> None:
        
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.parser = argparse.ArgumentParser(description="Predictor")
        
        self.X_train = pd.read_csv(
            self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(
            self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(
            self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)
        
        self.vectorizer, self.X_train, self.X_test, self.y_train, self.y_test, self.label_to_id, self.id_to_label = \
                                                                            vectorize(self.X_train, 
                                                                                      self.X_test, 
                                                                                      self.y_train, 
                                                                                      self.y_test)

    def predict(self, pred_args=None) -> bool:

        args_model = pred_args['model']
        args_tests = pred_args['tests']
        result = ''
        try:
            classifier = pickle.load(
                open(self.config[args_model]["path"], "rb"))
        except FileNotFoundError:
            sys.exit(1)
        
        if args_tests == "func":
            tests_path = os.path.join(os.getcwd(), "tests")
            exp_path = os.path.join(os.getcwd(), "experiments")
            for test in os.listdir(tests_path):
                with open(os.path.join(tests_path, test)) as f:
                    try:
                        data = json.load(f)
                        
                        X = self.vectorizer.transform([data['text']])
                        y = [(self.label_to_id[data['label']])]
                        
                        score = classifier.score(X, y)
                        print(f'{args_model} has {score} score')
                        result += f'{args_model} has {score} score \n'
                    except Exception:
                        sys.exit(1)
                    exp_data = {
                        "model": args_model,
                        "model params": dict(self.config.items(args_model)),
                        "tests": args_tests,
                        "score": str(score),
                        "X_test path": self.config["SPLIT_DATA"]["x_test"],
                        "y_test path": self.config["SPLIT_DATA"]["y_test"],
                    }
                    date_time = datetime.fromtimestamp(time.time())
                    str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                    exp_dir = os.path.join(exp_path, f'exp_{test[:6]}_{str_date_time}')
                    os.mkdir(exp_dir)
                    with open(os.path.join(exp_dir,"exp_config.yaml"), 'w') as exp_f:
                        yaml.safe_dump(exp_data, exp_f, sort_keys=False)
                    shutil.copy(os.path.join(os.getcwd(), "logfile.log"), os.path.join(exp_dir,"exp_logfile.log"))
                    shutil.copy(self.config[args_model]["path"], os.path.join(exp_dir,f'exp_{args_model}.sav'))
        if pred_args is not None:
            return result
        return True


app = Flask(__name__)

cmd = 'python ' + os.path.join(os.getcwd(), "src/preprocess.py")
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
# cmd = 'python ' + os.path.join(os.getcwd(), "src/train.py")
# p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

@app.route('/', methods= ["GET", "POST"])
def hello_world():
    if request.method == 'POST':
        model = str(request.values.get('model'))
    else:
        model = "LOG_REG"
    test = 'func'
    pred_args = {'model': model, 'test': test}
    # predictor = WebPredictor()
    # message=predictor.predict(pred_args)
    message = os.getcwd()
    config = configparser.ConfigParser()
    config.read("config.ini")
    message += ' ' + config["SPLIT_DATA"]["X_train"]
    message += ' ' + config["SPLIT_DATA"]["y_train"]
    message += ' ' + config["SPLIT_DATA"]["y_test"]
    message += ' ' + config["SPLIT_DATA"]["X_test"]
    return message

if __name__ == '__main__':
    app.run(debug=True)
