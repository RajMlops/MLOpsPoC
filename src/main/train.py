import sys
import logging
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from src.utils.config import get_default
from src.utils.data_utils import *
from src.main.preprocess import pre_processing
from src.utils.model_utils import save_model

logger = logging.getLogger(__name__)
PARAM_GRID = get_default('param_grid', 'param_grid')


def main():
    """Load the train data"""
    train_data_df = train_load_data()
    train_data_df = pre_processing(train_data_df)
    y_data = train_data_df['Transported']
    col = train_data_df.columns
    col = col.delete(6)
    print(col)
    x_data = train_data_df[col]

    """Split the data"""
    x_train, x_test, y_train, y_test = stratified_split(x_data, y_data)

    """Check the best classifier model"""
    clf = LazyClassifier()
    model, predictions = clf.fit(x_train, x_test, y_train, y_test)
    logger.info('Model details: {}'.format(model))

    """ LGBMClassifier has better performance compared to other classifiers """
    lgbm = lgb.LGBMClassifier(random_state=5)
    logger.info('Training model')
    lgbm.fit(x_train, y_train)
    pred = lgbm.predict(x_test)
    lgbm.get_params()
    logger.info('LGBMClassifier train Score: {}'
                .format(lgbm.score(x_train, y_train)))
    logger.info('LGBMClassifier test Score: {}'
                .format(lgbm.score(x_test, y_test)))

    """Check if the model does not overfit"""
    y_pred_train = lgbm.predict(x_train)

    """Check the accuracy"""
    logger.info('Accuracy Score: {}'
                .format(accuracy_score(y_train, y_pred_train)))

    """Hyperparameter tuning"""
    rscv = RandomizedSearchCV(estimator=lgbm,
                              param_distributions=PARAM_GRID,
                              scoring='accuracy')
    search = rscv.fit(x_train, y_train)
    logger.info('best params: {}'.format(search.best_params_))
    logger.info('best params: {}'.format(search.best_score_))

    lgbmhp = lgb.LGBMClassifier(max_bin=250,
                                learning_rate=0.03,
                                num_iterations=150,
                                min_gain_to_split=1,
                                max_depth=20)

    """Model Training"""
    lgbmhp.fit(x_train, y_train)

    """ModelDetails """
    prediction = lgbmhp.predict(x_test)
    confusionmatrix = confusion_matrix(y_test, prediction)
    sns.heatmap(confusionmatrix, annot=True)

    """Save Model"""
    save_model(lgbmhp)


if __name__ == '__main__':
    main()
