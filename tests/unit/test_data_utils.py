import unittest
import sys
import pandas as pd
sys.path.append('../..')
sys.path.append('.')
#import src.utils.data_utils as data_utils  # noqa: E402
#sys.path.append("/mnt/c/Users/Raj/Desktop/poc")
from src.utils.data_utils import stratified_split


X = ['0001_01','0002_01','0003_01','0003_02','0004_01','0005_01','0006_01','0006_02','0007_01','0008_01']
Y = ['FALSE','TRUE','FALSE','FALSE','TRUE','TRUE','TRUE','TRUE','TRUE','TRUE']

x = pd.DataFrame(X)
y = pd.DataFrame(Y)

class DataUtilsUnitTest(unittest.TestCase):
    def test_stratified_split(self):
        x_train,x_test,y_train,y_test = stratified_split(x,y)
        self.assertEqual(x_train.shape[0], 6, '6 observations in training set')
        self.assertEqual(x_test.shape[0], 4, '4 observations in testing set')
        self.assertEqual(y_train.shape[0], 6, '6 observations in training set')
        self.assertEqual(y_test.shape[0], 4, '4 observations in testing set')

if __name__ == "__main__":
    unittest.main()
