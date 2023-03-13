import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from src.utils.config import get_default
from src.utils.data_utils import *


def main():
    print("data loading starting...")
    # Create a histogram
    fig = px.bar(x=["a", "b", "c"], y=[1, 3, 2])
    fig.show()
    df = train_load_data()
    pio.renderers.default = "notebook"
    fig = px.histogram(df,
                       x='HomePlanet',
                       color='Transported',
                       barmode='group')
    fig = px.histogram(df,
                       x='CryoSleep',
                       color='Transported',
                       barmode='group')
    fig = px.histogram(df,
                       x='Destination',
                       color='Transported',
                       barmode='group')
    fig = px.histogram(df,
                       x='VIP',
                       color='Transported',
                       barmode='group')
    fig.show()


if __name__ == '__main__':
    main()
