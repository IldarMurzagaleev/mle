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
import traceback
import yaml
from src.logger import Logger

from src import utils

SHOW_LOG = True


class Predictor():
    """
     Class to implement prediction of different models
    """
    def __init__(self) -> None:
        """
        __init__ method which sets prediction process parameters and 
        carries out the preprocessing process with vectorizer
        """
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        self.parser = argparse.ArgumentParser(description="Predictor")
        self.parser.add_argument("-m",
                                 "--model",
                                 type=str,
                                 help="Select model",
                                 required=True,
                                 default="LOG_REG",
                                 const="LOG_REG",
                                 nargs="?",
                                 choices=["LOG_REG", "RAND_FOREST", "MULTI_NB"])
        self.parser.add_argument("-t",
                                 "--tests",
                                 type=str,
                                 help="Select tests",
                                 required=True,
                                 default="smoke",
                                 const="smoke",
                                 nargs="?",
                                 choices=["smoke", "func"])
        self.X_train = pd.read_csv(
            self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(
            self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(
            self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)
        
        self.vectorizer, self.X_train, self.X_test, self.y_train, self.y_test, self.label_to_id, self.id_to_label = \
                                                                            utils.vectorize(self.X_train, 
                                                                                      self.X_test, 
                                                                                      self.y_train, 
                                                                                      self.y_test)
        self.log.info("Predictor is ready")

    def predict(self, pred_args=None) -> bool:
        """
        Class method that implements text label prediction with selected model and parameters

        Args:
            pred_args (dict):  dictionary with model parameters when called from a web application 
            (rather than the command line) (default None)

        Returns:
            boolean execution success flag in case calling from command line or 
            text result of prediction 
        """
        if pred_args is None:
            args = self.parser.parse_args()
            args_model = args.model
            args_tests = args.tests
        else:
            args_model = pred_args['model']
            args_tests = pred_args['tests']
        result = ''
        try:
            classifier = pickle.load(
                open(self.config[args_model]["path"], "rb"))
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if args_tests == "smoke":
            try:
                score = classifier.score(self.X_test, self.y_test)
                print(f'{args_model} has {score} score')
                result = f'{args_model} has {score} score'
            except Exception:
                self.log.error(traceback.format_exc())
                sys.exit(1)
            self.log.info(
                f'{self.config[args_model]["path"]} passed smoke tests')
            

        elif args_tests == "func":
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
                        result += f'{args_model} has {score} score for {test} <br>'
                    except Exception:
                        self.log.error(traceback.format_exc())
                        sys.exit(1)
                    self.log.info(
                        f'{self.config[args_model]["path"]} passed func test {f.name}')
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


if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict()
