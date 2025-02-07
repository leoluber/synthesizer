import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def plot_correlations():

    """
        Plotting coorelations between molecule properties and the overall
        antisolvent effect (measured by sub4ML area)
    """

    # read areas df
    areas = pd.read_csv("data//processed//areas.csv")

    # read molecule encodings
