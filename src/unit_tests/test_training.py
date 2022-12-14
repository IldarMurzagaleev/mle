import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from train import MultiModel, main_block

config = configparser.ConfigParser()
config.read("config.ini")


class TestMultiModel(unittest.TestCase):

    def setUp(self) -> None:
        self.multi_model = MultiModel()

    def test_log_reg(self):
        self.assertEqual(self.multi_model.log_reg(), True)

    def test_rand_forest(self):
        self.assertEqual(self.multi_model.rand_forest(), True)
    
    def test_rand_forest_config(self):
        self.assertEqual(self.multi_model.rand_forest(use_config=True), True)

    def test_multi_nb(self):
        self.assertEqual(self.multi_model.multi_nb(), True)
    
    def test_main_block(self):
        self.assertEqual(main_block(), True)


if __name__ == "__main__":
    unittest.main()
