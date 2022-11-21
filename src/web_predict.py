from flask import Flask, render_template, request, jsonify
import os
import configparser
import subprocess
from src.predict import Predictor
from src.utils import vectorize

app = Flask(__name__)

cmd = 'python ' + os.path.join(os.getcwd(), "src/preprocess.py") + "; "
cmd +=  'python ' + os.path.join(os.getcwd(), "src/train.py") + "; "
cmd +=  'python ' + os.path.join(os.getcwd(), "src/predict.py") + " -m LOG_REG -t func ; "
cmd += 'coverage run ' + os.path.join(os.getcwd(), "src/unit_tests/test_preprocess.py") + "; " 
cmd += 'coverage run -a ' +  os.path.join(os.getcwd(), "src/unit_tests/test_training.py") + '; coverage report -m'
p = subprocess.Popen(['/bin/bash', '-c', cmd])


@app.route('/', methods= ["GET", "POST"])
def hello_world():
    if request.method == 'POST':
        model = str(request.values.get('model'))
    else:
        model = "LOG_REG"
    test = 'func'
    pred_args = {'model': model, 'tests': test}

    # message = os.getcwd() + " " + model + " "
    # config = configparser.ConfigParser()
    # config.read("config.ini")
    # message += ' ' + config["SPLIT_DATA"]["X_train"]
    # message += ' ' + config["SPLIT_DATA"]["y_train"]
    # message += ' ' + config["SPLIT_DATA"]["y_test"]
    # message += ' ' + config["SPLIT_DATA"]["X_test"]
    # message += ' ' + config["LOG_REG"]["path"]
    predictor = Predictor()
    message=predictor.predict(pred_args)
    return message

if __name__ == '__main__':
    app.run(debug=True)
