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
            fig = explore_Parameterspace(TwoDFrame, 
                    title = level,
                    cmap = cmap,
                    norm = cl.Normalize(vmin=0, vmax=vmax),
                    vmin=vmin,
                    vmax=vmax)
            fig.savefig(SAVE_PATH + '/' + level.strip('<>') + `save_name` + '.jpg' , format='jpg')


def explore_Parameterspace(TwoDFrame, title="",
                           cmap='RdBu', norm=None, vmin=None, vmax=None):
    """
    Explore variables in a 2-dim Parameterspace

    Parameters
    ----------
    TwoDFrame : 2D pandas.DataFrame with index and column names
        The data to plot
    title : string
        Title of the plot (Default: "")
    cmap : string
        Colormap to use (Default: "RdBu")
    vmin : float
        Minimum value of the colormap (Default: None)
    vmax : float
        Maximum vlaue of the colormap (Defualt: None)

    Examples
    --------
    >>> import init_data
    >>> data = init_data.get_Data("phi")
    >>> explore_Parameterspace(data.unstack(level="deltaT")["<safe>",0.5].
    >>>                        unstack(level="phi"))
    """

    xparams = TwoDFrame.columns.values
    yparams = TwoDFrame.index.values
    values = TwoDFrame.values

    X, Y = _create_meshgrid(xparams, yparams)
    fig = plt.figure()
    c = plt.pcolormesh(X, Y, values, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    plt.colorbar(c, orientation="vertical")
    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))
    plt.xlabel(TwoDFrame.columns.name)
    plt.ylabel(TwoDFrame.index.name)
    plt.title(title)
    plt.tight_layout()

    return fig 

# TODO: Explore Parameterspace3D
# see: http://matplotlib.org/examples/mplot3d/contour3d_demo3.html
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# ax.plot_surface(X, Y, values, rstride=1, cstride=1, alpha=0.3, cmap=cm.PiYG)
# cset = ax.contour(X, Y, values, zdir='y', offset=0.9, cmap=cm.Blues)
# cset = ax.contour(X, Y, values, zdir='z', offset=0.0, cmap=cm.Blues)
# ax.set_xlabel("tau")
# ax.set_ylabel("sigma")


def _create_meshgrid(x, y):
    """
    Create a meshgrid out of the array-like types x and y. We assume that x and
    y are equally spaced. The funciton positions the values of x and y into the
    middle of the return value.

    Parameters
    ----------
    x : 1D array like
        The x values
    y : 1D array like
        The y values

    Returns
    -------
    meshgrid : 2D np.array
        widened meshgrid of x and y

    Example
    -------
    >>> _create_meshgrid([1,2], [10, 12])
    >>> [array([[ 0.5,  1.5,  2.5],
                [ 0.5,  1.5,  2.5],
                [ 0.5,  1.5,  2.5]]),
         array([[  9.,   9.,   9.],
                [ 11.,  11.,  11.],
                [ 13.,  13.,  13.]])]
    """
    x = np.array(x)
    y = np.array(y)

    def broaden_grid(x):
        dx = x[1:] - x[:-1]
        xx = np.concatenate(([x[0]], x))
        dxx = np.concatenate(([-dx[0]], dx, [dx[-1]]))
        return xx+dxx/2.

    X = broaden_grid(x)
    Y = broaden_grid(y)

    return np.meshgrid(X, Y)

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
