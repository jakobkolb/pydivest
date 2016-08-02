import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import itertools as it
import matplotlib.colors as cl
import matplotlib.cm as cm
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
    
    parameter_levels = [list(p.values) for p in data.index.levels[2:]]
    parameter_level_names = [name for name in data.index.names[2:]]
    levels = tuple(i+2 for i in range(len(parameter_levels)))

    parameter_combinations = list(it.product(*parameter_levels))
    
    for p in parameter_combinations:
        d_slice = data.xs(key=p, level=levels).dropna()
        save_name = zip(parameter_level_names, p)

        for level in list(d_slice.unstack().columns.levels[0]):
            TwoDFrame = d_slice[level].unstack()
            vmax = np.nanmax(TwoDFrame.values)
            vmin = np.nanmin(TwoDFrame.values)
            TwoDFrame.replace(np.nan, np.inf)
            cmap = cm.get_cmap('RdBu')
            cmap.set_under('yellow')
            cmap.set_over('black')
            print level, vmin, vmax
            fig = ev.explore_Parameterspace(TwoDFrame, 
                    title = level,
                    cmap = cmap,
                    norm = cl.Normalize(vmin=0, vmax=vmax),
                    vmin=vmin,
                    vmax=vmax)
            fig.savefig(SAVE_PATH + '/' + level.strip('<>') + `save_name` + '.jpg' , format='jpg')

def plot_trj_grid(SAVE_PATH, NAME):

    data = np.load(SAVE_PATH+NAME)
    parameter_levels = [list(p.values) for p in data.index.levels[2:-2]]
    parameter_level_names = [name for name in data.index.names[2:-2]]
    levels = tuple(i+2 for i in range(len(parameter_levels)))
    parameter_combinations = list(it.product(*parameter_levels))
    
    for p in parameter_combinations:
        d_slice = data.xs(key=p, level=levels).dropna()
        save_name = zip(parameter_level_names, p)

if __name__ == "__main__":
    
    if getpass.getuser() == "kolb":
        SAVE_PATH = "/home/kolb/Divest_Experiments/divestdata/X1/"
    elif getpass.getuser() == "jakob":
        SAVE_PATH = "/home/jakob/PhD/Project_Divestment/Implementation/divestdata/X1/"
        
    NAME = "experiment_testing_tau_vs_phi"
    INDEX = {0: "tau", 1: "phi"}

    plot_tau_phi(SAVE_PATH, NAME)
