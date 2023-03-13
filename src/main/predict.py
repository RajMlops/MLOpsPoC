import sys
import logging
import pandas as pd
from src.utils.config import get_default
from src.utils.data_utils import *
from src.utils.model_utils import *
from src.main.preprocess import pre_processing

logger = logging.getLogger(__name__)


def main(**kwargs):
    lgbm = load_model()
    df = [
         {
            "PassengerId": "0013_01",
            "HomePlanet": "Earth",
            "CryoSleep": "TRUE",
            "Cabin": "G/3/S",
            "Destination": "TRAPPIST-1e",
            "Age": 27,
            "VIP": "FALSE",
            "RoomService": 0,
            "FoodCourt": 0,
            "ShoppingMall": 0,
            "Spa": 0,
            "VRDeck": 0,
            "Name": "Nelly Carsoning"
            },
         {
            "PassengerId": "0018_01",
            "HomePlanet": "Earth",
            "CryoSleep": "FALSE",
            "Cabin": "F/4/S",
            "Destination": "PSO J318.5-22",
            "Age": 19,
            "VIP": "TRUE",
            "RoomService": 0,
            "FoodCourt": 9,
            "ShoppingMall": 0,
            "Spa": 2823,
            "VRDeck": 0,
            "Name": "Lerome Peckers"
            },
         {
            "PassengerId": "0019_01",
            "HomePlanet": "Mars",
            "CryoSleep": "TRUE",
            "Cabin": "C/0/S",
            "Destination": "55 Cancri e",
            "Age": 31,
            "VIP": "FALSE",
            "RoomService": 0,
            "FoodCourt": 0,
            "ShoppingMall": 0,
            "Spa": 0,
            "VRDeck": 0,
            "Name": "Sabih Unhearfus"
            },
         {
            "PassengerId": "0021_01",
            "HomePlanet": "Europa",
            "CryoSleep": "FALSE",
            "Cabin": "C/1/S",
            "Destination": "TRAPPIST-1e",
            "Age": 38,
            "VIP": "TRUE",
            "RoomService": 0,
            "FoodCourt": 6652,
            "ShoppingMall": 0,
            "Spa": 181,
            "VRDeck": 585,
            "Name": "Meratz Caltilter"
            }
         ]
    test_data_dataframe = pd.DataFrame(df)

    sub = pd.DataFrame(test_data_dataframe['PassengerId'])
    new_test_data_df = pre_processing(test_data_dataframe)

    prediction = lgbm.predict(new_test_data_df)

    print(prediction)
    logger.info('Predicted class: {}'.format(prediction[0]))
    return prediction[0]


if __name__ == '__main__':
    main()
