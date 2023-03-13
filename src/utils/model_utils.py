from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
import pickle
import lightgbm as lgb
from src.utils.config import get_default
from src.utils.data_utils import *


def save_model(model: lgb.LGBMClassifier) -> None:
    """Save latest trained model to models folder."""
    print('saving model')
    path = 'src/resources/models/'
    filename = 'LGBMClassifier_model.pkl'
    pickle.dump(model, open(path + filename, 'wb'))


def load_model() -> lgb.LGBMClassifier:
    """Load latest trained model from models folder."""
    print('loading model')
    path = 'src/resources/models/'
    filename = 'LGBMClassifier_model.pkl'
    model = pickle.load(open(path + filename, 'rb'))
    return model
