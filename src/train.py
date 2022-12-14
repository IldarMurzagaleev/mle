import configparser
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import traceback

from src.logger import Logger
from src.utils import *

SHOW_LOG = True


class MultiModel():
    """
     Class to implement training and saving different models
    """ 
    def __init__(self) -> None:
        """
        __init__ method which sets training process parameters and 
        carries out the preprocessing process with vectorizer
        """
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
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
        self.project_path = os.path.join(os.getcwd(), "experiments")
        self.log_reg_path = os.path.join(self.project_path, "log_reg.sav")
        self.rand_forest_path = os.path.join(self.project_path, "rand_forest.sav")
        self.multi_nb_path = os.path.join(self.project_path, "multi_nb.sav")
        self.log.info("MultiModel is ready")
    
    def log_reg(self, predict=False) -> bool:
        """
        Class method which train and save logistic regression model

        Args:
            predict (bool): boolean train / predict mode flag (True if training mode) 

        Returns:
            boolean execution success flag
        """
        classifier = LogisticRegression()
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(accuracy_score(self.y_test, y_pred))
        params = {'path': self.log_reg_path}
        return self.save_model(classifier, self.log_reg_path, "LOG_REG", params)
    
    def multi_nb(self, predict=False) -> bool:
        """
        Class method which train and save multinomial bayes model

        Args:
            predict (bool): boolean train / predict mode flag (True if training mode) 

        Returns:
            boolean execution success flag
        """
        classifier = MultinomialNB()
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(accuracy_score(self.y_test, y_pred))
        params = {'path': self.multi_nb_path}
        return self.save_model(classifier, self.multi_nb_path, "MULTI_NB", params)
    
    def rand_forest(self, use_config=False, n_trees=200, max_depth=100, criterion="gini", predict=False) -> bool:
        """
        Class method which train and save random forest model

        Args:
            use_config (bool): boolean using config file flag (True if use config file)
            predict (bool): boolean train / predict mode flag (True if training mode)
            n_trees (int): number of trees in model (default 200)
            max_depth (int): the maximum depth of the tree
            criterion {???gini???, ???entropy???, ???log_loss???}: the function to measure the quality of a split (default=???gini???)


        Returns:
            boolean execution success flag
        """
        if use_config:
            try:
                classifier = RandomForestClassifier(
                    n_estimators=self.config.getint("RAND_FOREST", "n_estimators"), max_depth=self.config.getint("RAND_FOREST", "max_depth"), criterion=self.config["RAND_FOREST"]["criterion"])
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config:{use_config}, no params')
                sys.exit(1)
        else:
            classifier = RandomForestClassifier(
                n_estimators=n_trees, max_depth=max_depth, criterion=criterion)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(accuracy_score(self.y_test, y_pred))
        params = {'n_estimators': n_trees,
                  'max_depth': max_depth,
                  'criterion': criterion,
                  'path': self.rand_forest_path}
        return self.save_model(classifier, self.rand_forest_path, "RAND_FOREST", params)

    def save_model(self, classifier, path: str, name: str, params: dict) -> bool:
        """
        Class method which save file with trained model

        Args:
            classifier: trained model
            path (str): path for saving file with model
            name (str): model name for config file
            params (dict): model parameter 

        Returns:
            boolean execution success flag
        """
        self.config[name] = params
        os.remove('config.ini')
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


def main_block():
    """
    Class method which execute all models training

    Args:

    Returns:
    boolean execution success flag
    """
    multi_model = MultiModel()
    multi_model.log_reg(predict=True)
    multi_model.rand_forest(predict=True)
    multi_model.multi_nb(predict=True)
    return True


if __name__ == "__main__":
    main_block()
