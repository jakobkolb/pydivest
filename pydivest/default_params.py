# Copyright (C) 2016-2018 by Jakob J. Kolb at Potsdam Institute for Climate
# Impact Research
#
# Contact: kolb@pik-potsdam.de
# License: GNU AGPL Version 3

import getpass
import os
from pathlib import Path
from pydivestparameters.rs_models.parameter_fit import ParameterFit

import pandas as pd


class ExperimentDefaults:
    """contains default values for experiment default parameters
    to keep them consistent
    """

    def __init__(self,
                 params='default',
                 alpha=2./3.,
                 gamma=1./8.,
                 chi=0.02,
                 **kwargs):
        """setting input parameter values

        Parameters
        ----------
        params: string
            one of [default, fitted] switch to change between default
            parameters or fitted parameters.
        alpha: float
            for fitting parameters, this can be a custom factor elasticity for
            labor
        gamma: float
            for fitting parameters, this can be a custom factor elasticity for
            knowledge in the clean sector
        chi: float
            for fitting parameters, this can be a custom depreciation rate for
            knowledge
            """

        input_params = {
            'b_c': 1.,
            'b_d': 4.,
            'e': 1.,
            'b_r0': 0.1,
            'kappa_c': .5,
            'kappa_d': .5,
            'pi': .5,
            'xi': .1,
            'd_k': 0.06,
            'd_c': 0.1,
            'phi': .5,
            'tau': 1.,
            'eps': 0.03,
            's': 0.23,
            'possible_cue_orders': [[0], [1]],
            'L': 1.,
            'C': 1.,
            'campaign': False,
            'learning': True,
            'interaction': 1,
            'test': False,
            'R_depletion': True
        }

        input_params_fitted = {
            'b_c':  22.94219556006435,
            'b_d':  1844.7328627029444,
            'e':    45048994159.98814,
            'b_r0': 1041159355.7928585,
            'kappa_c': 1./3.,
            'kappa_d': 1./3.,
            'pi': 2./3,
            'xi': 1/8,
            'd_k': 0.056818181818181816,
            'd_c': 0.02,
            'phi': .5,
            'tau': 1.,
            'eps': 0.03,
            's': 0.25,
            'possible_cue_orders': [[0], [1]],
            'campaign': False,
            'learning': True,
            'interaction': 1,
            'test': False,
            'R_depletion': True,
            'G_0': 1584455.821037092,
            'G': 1155584.3903578299,
            'C': 779787753829056.2,
            'K_c0': 311922705312234.7,
            'K_d0': 2041863385150630.2,
            'L': 3288420945.5,
        }

        # check which set of parameters to use
        if params == 'default':
            self.input_params = input_params
        elif params == 'fitted':
            if alpha != 2./3. or gamma != 1./8. or chi != 0.02:
                param_model = ParameterFit(alpha=alpha,
                                           gamma=gamma,
                                           chi=chi)
                param_model.fit_params()
                self.input_params = param_model.get_params()
                for key, value in input_params_fitted.items():
                    if key not in self.input_params.keys():
                        self.input_params[key] = value
            else:
                self.input_params = input_params_fitted

        for key, val in kwargs.items():
            if key not in self.input_params.keys():
                raise KeyError(f'{key} is not a valid input parameter')
            else:
                self.input_params[key] = val
        if params == 'default':
            self.calculate_timing(t_g=100, t_a=None)


    def calculate_timing(self, t_g, t_a=None):
        """calculate parameters of the economy to reproduce given timing
        """

        e = self.input_params['e']
        delta = self.input_params['d_k']
        s = self.input_params['s']
        L = self.input_params['L']
        b_d = self.input_params['b_d']
        b_R = self.input_params['b_r0']

        G_0 = t_g * (s * L * b_d**2) / (e * delta) / (1. - b_R / e)

        self.input_params['G_0'] = G_0

        if t_a is not None:
            self.input_params['tau'] = t_a / (1 - self.input_params['phi'])

    def get_social_dynamics_timescale(self):
        """return timescale for social dynamics given that one opinion
        dominates all others
        """
        t_sd = self.input_params['tau'] * (1 - self.input_params['phi'])
        return t_sd

    def get_dirty_capital_timescale(self):
        """return timescale for dirty capital accumulation
        """
        # ToDo: Doublecheck this!!
        t_dc = 1 / (self.input_params['d_k']*self.input_params['pi'])
        return t_dc

    def get_clean_capital_timescale(self):
        """return timescale for clean capital accumulation
        """
        # ToDo: Doublecheck this!!
        t_cc = 1/(self.input_params['d_k']*self.input_params['pi']*(1-self.input_params['xi']))

        return t_cc


class ExperimentRoutines:
    def __init__(self,
                 run_func,
                 param_combs,
                 test,
                 subfolder,
                 data_folder='output_data',
                 local_user='jakob',
                 cluster_user='kolb'):
        """
        set input/output paths

        Parameters:
        -----------
        test: bool
            switch for testing, triggers writing into extra test output sub
            folder
        subfolder: str
            name of subfolder for specific experiment
        data_folder: str
            name of data output folder
        local_user: str
            user name on local machine (for testing purposes)
        cluster_user: str
            user name on production machine - triggers writing of raw output to
            temporary file storage.
        """

        self.run_func = run_func

        respath = os.path.dirname(
            os.path.realpath(__file__)) + f"/../{data_folder}"

        if getpass.getuser() == local_user:
            tmppath = respath
        elif getpass.getuser() == cluster_user:
            tmppath = f"/p/tmp/{cluster_user}/Divest_Experiments"
        else:
            tmppath = "./"

        # make sure, testing output goes to its own folder:

        test_folder = ['', 'test_output/'][int(test)]

        # check if cluster or local and set paths accordingly
        self.save_path_raw = f"{tmppath}/{test_folder}{subfolder}/"
        self.save_path_res = f"{respath}/{test_folder}{subfolder}/"

        self.run_func_keywords = self._get_function_keywords()
        self.run_func_output = self._runfunc_dummy_output(run_func,
                                                             param_combs)

    def _get_function_keywords(self, function=None):
        """create dict of keywords for function keyword parameters

        Parameters:
        -----------
        function: callable
            function from which to extract keywords
        """
        if function is None:
            function = self.run_func

        if not callable(function):
            raise ValueError('function must be callable')
        else:
            index = {
                i: function.__code__.co_varnames[i]
                for i in range(function.__code__.co_argcount)
            }

        return index

    def get_paths(self):
        """return temporary and result save paths for this helper instance

        format is {home_path}/{test_folder}{subfolder}/

        Returns:
        --------
        save_path_raw: str
            path to temprary file storage (fast write, high volume, no backup)
        save_path_res: str
            path to result storage in user home folder
        """

        return self.save_path_raw, self.save_path_res

    def _runfunc_dummy_output(self, run_func, param_combs):
        """get run func dummy output.

        If there is some existing dummy output, load it
        if not, create new dummy output and save it

        Parameters:
        -----------
        param_combs: list
            list of parameter combinations for this experiment
        run_func: executable
            run func for this experiment
        """

        if not Path(self.save_path_raw).exists():
            os.makedirs(self.save_path_raw, exist_ok=True)
        try:
            run_func_output = pd.read_pickle(self.save_path_raw + 'rfof.pkl')
        except:
            params = list(param_combs[0])
            params[-1] = True
            run_func_output = run_func(*params)[1]
            with open(self.save_path_raw + 'rfof.pkl', 'wb') as dmp:
                pd.to_pickle(run_func_output, dmp)

        return run_func_output

    # define pp function for trajectories

    def _pp_data(self,
                 *args,
                 table_id=None,
                 operator='mean'):

        from pymofa.safehdfstore import SafeHDFStore

        query = ''
        for keyword, value in zip(self.run_func_keywords.values(), args):
            query += f'{keyword}={value}&'
        query = query[:-1]

        print(query)

        # set table query

        if table_id is None:
            table_queries = ['dat']
        else:
            table_queries = [f'dat_{did}' for did in table_id]

        dfs = []

        data_path = self.save_path_raw[:-1] + '.h5'
        print(data_path)

        for tq in table_queries:
            with SafeHDFStore(data_path) as store:
                trj = store.select(tq, where=query)

            # set operator to perform on data

            if operator == 'mean':
                df_out = trj.groupby(level=-1).mean()
            elif operator == 'std':
                df_out = trj.groupby(level=-1).std()
            elif operator == 'collect':
                # if the dataframe has acolumn called sample_id, use it to
                # store the sample id data that would otherwise go lost.
                if 'sample_id' in trj.columns:
                    trj.index = trj.index.droplevel(
                        list(self.run_func_keywords.values()))
                    trj.drop('sample_id', axis=1, inplace=True)
                    trj = trj.reset_index(level=['sample'])
                    trj.rename(columns={'sample': 'sample_id'},
                               inplace=True)
                else:
                    trj.index = trj.index.droplevel(
                        list(self.run_func_keywords.values()) + ['sample'])
                df_out = trj
            else:
                raise ValueError(f'operator {operator} not implemented')
            dfs.append(df_out)

        output_list = self.run_func_output

        for did, df in zip(table_id, dfs):
            output_list[did] = df

        return 1, output_list

    def get_pp_function(self, table_id=None, operator='mean'):
        def wrapper(*args, **kwargs):
            res = self._pp_data(*args,
                                table_id=table_id,
                                operator=operator,
                                **kwargs)

            return res

        return wrapper
