import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import getpass
from pymofa import experiment_visualization as ev

def plot_tau_phi(SAVE_PATH, NAME):
    """
    Loads the dataframe NAME of postprocessed data
    from the location given by SAVE_PATH
    and plots a heatmap for each of the layers of data
    in it.

    Parameters:
    -----------
    SAVE_PATH : string
        the location where the processed data is saved
    NAME : string
        the name of the datafile containing the processed
        data
    """

    data = np.load(SAVE_PATH + NAME)

    print data.unstack().columns.levels[0]

    for level in list(data.unstack().columns.levels[0]):

        print level

        fig = ev.explore_Parameterspace(data.unstack()[level], title = level)
        fig.savefig(SAVE_PATH + '/' + level.strip('<>') + '.jpg' , format='jpg')



if __name__ == "__main__":
    
    if getpass.getuser() == "kolb":
        SAVE_PATH = "/home/kolb/Divest_Experiments/divestdata/X1/"
    elif getpass.getuser() == "jakob":
        SAVE_PATH = "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/X1/"
        
    NAME = "experiment_testing_tau_vs_phi"
    INDEX = {0: "tau", 1: "phi"}

    plot_tau_phi(SAVE_PATH, NAME)
