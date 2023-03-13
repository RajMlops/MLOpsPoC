import unittest
import sys
import warnings
import lightgbm as lgb
sys.path.append('../..')
sys.path.append('.')
import src.utils.model_utils as model_utils 


def ignore_warnings(test_func):
    """Decorator to ignore warnings in unittest"""
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)
    return do_test


class ModelUtilsUnitTest(unittest.TestCase):
    @ignore_warnings
    def test_save_load_model(self):
        model_utils.save_model(lgb.LGBMClassifier)
        loaded_model = model_utils.load_model()
        self.assertEqual(lgb.LGBMClassifier, loaded_model)


if __name__ == "__main__":
    unittest.main()
