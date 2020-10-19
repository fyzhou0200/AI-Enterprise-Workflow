#!/usr/bin/env python
"""
model tests
"""


import unittest

## import model specific functions and variables
from model import *
SAVED_MODEL = os.path.join(MODEL_DIR,"test-{}-{}.joblib".format('all',re.sub("\.","_",str(MODEL_VERSION))))

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """

    def test_01_train(self):
        """
        test the train functionality
        """
        data_dir = os.path.join("data","cs-train")
        ## train the model
        model_train(data_dir,test=True)
        self.assertTrue(os.path.exists(SAVED_MODEL))

    def test_02_load(self):
        """
        test the train functionality
        """

        ## load the model
        print("LOADING MODELS")
        all_data, all_models = model_load()
        print("... models loaded: ",",".join(all_models.keys()))

        for tag, model in all_models.items():
            self.assertTrue('predict' in dir(model))
            self.assertTrue('fit' in dir(model))


    def test_03_predict(self):
        """
        test the predict function input
        """

        ## load the model
        print("LOADING MODELS")
        all_data, all_models = model_load()
        print("... models loaded: ",",".join(all_models.keys()))

        country='all'
        year='2018'
        month='01'
        day='05'
        result = model_predict(country,year,month,day)
        print(result)

        y_pred = result['y_pred']
        self.assertTrue(y_pred.dtype==np.float64)



### Run the tests
if __name__ == '__main__':
    unittest.main()
