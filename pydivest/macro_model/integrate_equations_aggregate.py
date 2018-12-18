"""Part of the pydivestment package
authored by Jakob J. Kolb (kolb@pik-potsdam.de)"""

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import odeint

try:
    from .integrate_equations import IntegrateEquations
except ModuleNotFoundError:
    from integrate_equations import IntegrateEquations


class IntegrateEquationsAggregate(IntegrateEquations):
    def __init__(self, adjacency=None, investment_decisions=None,
                 investment_clean=None, investment_dirty=None,
                 tau=0.8, phi=.7, eps=0.05,
                 b_c=1., b_d=1.5, s=0.23, d_k=0.06, d_c=0.06,
                 b_r0=1., e=10,
                 pi=0.5, kappa_c=0.5, kappa_d=0.5, xi=1. / 8.,
                 L=100., G_0=3000, C=1,
                 R_depletion=True,
                 interaction=1, crs=True, test=False,
                 **kwargs):
        """
        integrate the approximate macro equations for the pydivest module in the aggregate
        formulation.
        Allows for non-constant returns to scale if crs is set to False
        and values for kappa_c and kappa_d are provided.

        Parameters
        ----------
        adjacency: ndarray
            Acquaintance matrix between the households. Has to be symmetric unweighted and without self loops.
        investment_decisions: list
            Initial investment decisions of households. Will be updated
            from their actual heuristic decision making during initialization
        investment_clean: list
            Initial household endowments in the clean sector
        investment_dirty: list
            Initial household endowments in the dirty sector
        i_tau: float
            Mean waiting time between household opinion updates
        i_phi: float
            Rewiring probability in the network adaptation process
        eps: float
            fraction of exploration events (noise) in the opinion formation process
        b_c: float
            Solow residual of the production function of the clean sector
        b_d: float
            Solow residual of the production function of the dirty sector
        s: float
            Savings rate of the households
        d_k: float
            Capital depreciation rate
        d_c: float
            Knowledge depreciation rate
        b_r0: float
            Resource cost factor
        e: float
            Resource efficiency in the dirty sector
        pi: float
            labor elasticity for both sectors
        kappa_c: float
            capital elasticity for the clean sector
        kappa_d:
            capital elasticity for the dirty sector
        xi:
            elasticity of the knowledge stock in the clean sector
        L: float
            Total labor (fixed)
        G_0: float
            Total initial resource stock
        C: float
            Total initial knowledge stock
        resource_depletion: bool
            Switch to turn resource depreciation on or off
        interaction: int
            Switch for different imitation probabilities.
            if 0: tanh(Wi-Wj) interaction,
            if 1: interaction as in Traulsen, 2010 but with relative differences
            if 2: (Wi-Wj)/(Wi+Wj) interaction.
            if 3: random imitation e.g. p_cd = p_dc = .5
        crs: bool
            switch for constant returns to scale. If True, values of kappa are ignored.
        """

        super().__init__(adjacency=adjacency, investment_decisions=investment_decisions,
                         investment_clean=investment_clean, investment_dirty=investment_dirty,
                         tau=tau, phi=phi, eps=eps,
                         pi=pi, kappa_c=kappa_c, kappa_d=kappa_d, xi=xi,
                         L=L, b_c=b_c, b_d=b_d, s=s, d_c=d_c, d_k=d_k,
                         b_r0=b_r0, e=e, G_0=G_0, C=C,
                         R_depletion=R_depletion, test=test, crs=crs, interaction=interaction)

        if test:
            def verboseprint(*args):
                for stuff in args:
                    print(stuff)
        else:
            def verboseprint(*args):
                pass

        self.verboseprint = verboseprint

        self.verboseprint('got superfluous keyword arguments')
        self.verboseprint(kwargs.keys())

        c = self.investment_decisions
        d = - self.investment_decisions + 1

        self.v_Kcc = sum(self.investment_clean * c)
        self.v_Kdc = sum(self.investment_dirty * c)
        self.v_Kcd = sum(self.investment_clean * d)
        self.v_Kdd = sum(self.investment_dirty * d)

        # aggreagate capital endowments of clean and dirty households
        # lower (first) index: capital type, upper (second) index: household type
        self.Kcc, self.Kcd, self.Kdc, self.Kdd = sp.symbols('K_c^c K_c^d K_d^c K_d^d', positive=True, real=True)

        # create list of symbols and names of all independent variables
        self.var_symbols = [self.x, self.y, self.z, self.Kcc, self.Kcd, self.Kdc, self.Kdd, self.C, self.G]
        self.var_names = ['x', 'y', 'z', 'K_c^c', 'K_c^d', 'K_d^c', 'K_d^d', 'C', 'G']

        self.independent_vars = {name: symbol for name, symbol in zip(self.var_names, self.var_symbols)}

        # define expected wealth as expected income
        self.subs1[self.Wc] = ((self.rc * self.Kcc + self.rd * self.Kdc + self.w * self.P * self.Nc / self.N)
                               / self.Nc).subs(self.subs1)
        self.subs1[self.Wd] = ((self.rc * self.Kcd + self.rd * self.Kdd + self.w * self.P * self.Nd / self.N)
                               / self.Nd).subs(self.subs1)

        if interaction == 0:
            raise ValueError('only interactions depending on relative differences of agent properties are'
                             'possible with a macroscopic approximation in aggregate quantities')

        # Define clean and dirty capital as weighted sums over aggregate endowments
        self.subs4[self.Kc] = self.Kcc + self.Kcd
        self.subs4[self.Kd] = self.Kdc + self.Kdd

        # Write down dynamic equations for the economic subsystem in terms of means of clean and dirty capital stocks
        # for clean and dirty households

        self.rhsECO_1 = sp.Matrix(
            [(self.rs * self.rc - self.delta_k) * self.Kcc + self.rs * self.rd * self.Kdc
             + self.rs * self.w * self.P * self.Nc / self.N,
             -self.delta_k * self.Kcd,
             -self.delta_k * self.Kdc,
             self.rs * self.rc * self.Kcd + (self.rs * self.rd - self.delta_k) * self.Kdd
             + self.rs * self.w * self.P * self.Nd / self.N,
             self.bc * self.Pc ** self.pi * self.Kc ** self.kappac * self.C ** self.xi - self.delta_c * self.C,
             -self.R])
        # Write down changes in means of capital stocks through agents'
        # switching of opinions and add them to the capital accumulation terms

        self.rhsECO_switch_1 = sp.Matrix([
            # change of clean capital owned by clean investors
            self.Kcd / self.Nd * self.dtNdc - self.Kcc / self.Nc * self.dtNcd,
            # change of clean capital owned by dirty investors
            self.Kcc / self.Nc * self.dtNcd - self.Kcd / self.Nd * self.dtNdc,
            # change in dirty capital owned by clean investors
            self.Kdd / self.Nd * self.dtNdc - self.Kdc / self.Nc * self.dtNcd,
            # change in dirty capital owned by dirty investors
            self.Kdc / self.Nc * self.dtNcd - self.Kdd / self.Nd * self.dtNdc,
            0,
            0])

        self.rhsECO_switch_2 = self.rhsECO_switch_1.subs(self.subs1)

        self.rhsECO_2 = self.rhsECO_1 + self.rhsECO_switch_2

        # In the economic system, substitute:
        # 1)primitive variables for dependent variables (subs2)
        # 2)dependent variables for system variables (subs3)

        self.rhsECO_3 = self.rhsECO_2.subs(self.subs1).subs(self.subs2).subs(self.subs3).subs(self.subs4)

        # In the PBP rhs substitute:
        # dependent variables for system variables
        # NOTE TO SELF: DO NOT WRITE TO PARENT CLASS VARIABLES. THIS WILL BACKFIRE, IF OTHER
        # CLASSES INHERIT FROM THE SAME PARENT!

        self.rhsPBP_1 = self.rhsPBP.subs(self.subs1)

        self.rhsPBP_2 = self.rhsPBP_1.subs(self.subs1).subs(self.subs2).subs(self.subs3).subs(self.subs4)

        # Combine dynamic equations of economic and social subsystem:

        self.rhs_raw = sp.Matrix([self.rhsPBP_2, self.rhsECO_3]).subs(self.subs1)

        # update dependent vars with specific approximation variables
        self.update_dependent_vars()
        # Set parameter values in rhs and dependent variables:
        self.set_parameters()

        self.m_trajectory = pd.DataFrame(columns=self.var_names)

        # dictionary for final state
        self.final_state = {}

    def list_initial_conditions(self):
        values = [self.v_x, self.v_y, self.v_z,
                  self.v_Kcc, self.v_Kcd,
                  self.v_Kdc, self.v_Kdd,
                  self.v_C, self.v_G]
        return {symbol: val for symbol, val in zip(self.var_symbols, values)}

    def run(self, t_max=100, t_steps=500):
        """
        run the model for a given time t_max and produce results in resolution t_steps
        Parameters
        ----------
        t_max: float
            upper limit of simulation time
        t_steps: int
            number of timesteps of result

        Returns
        -------
        rval: int
            positive, if the simulation succeeded.
        """
        self.p_t_max = t_max

        if t_max > self.v_t:
            self.verboseprint('integrating equations from t={} to t={}'.format(self.v_t, t_max))

            t = np.linspace(self.v_t, t_max, t_steps)

            initial_conditions = [self.v_x, self.v_y, self.v_z,
                                  self.v_Kcc, self.v_Kcd,
                                  self.v_Kdc, self.v_Kdd,
                                  self.v_C, self.v_G]

            trajectory = odeint(self.dot_rhs, initial_conditions, t)

            df = pd.DataFrame(trajectory, index=t, columns=self.var_names)
            self.m_trajectory = pd.concat([self.m_trajectory, df])

            # update aggregated variables:
            (self.v_x, self.v_y, self.v_z,
             self.v_Kcc, self.v_Kcd,
             self.v_Kdc, self.v_Kdd,
             self.v_C, self.v_G) = trajectory[-1]

            self.v_c = self.C / self.p_n
            self.v_g = self.G / self.p_n
            self.v_t = t_max

        elif t_max <= self.v_t:
            self.verboseprint('upper time limit is smaller than system time', self.v_t)

        return 1

    def get_aggregate_trajectory(self):

        df = self.m_trajectory

        df['N_c over N'] = .5 * (df['x'] + 1)
        df['[cc] over M'] = .5 * (1 + df['y'] - df['z'])
        df['[cd] over M'] = df['z']

        return df

    def get_mean_trajectory(self):
        """return a mock mean trajectory with correct shape but containing zeros"""

        columns = ['x', 'y', 'z', 'mu_c^c', 'mu_d^c', 'mu_c^d', 'mu_d^d', 'c', 'g']
        index = self.m_trajectory.index

        df = pd.DataFrame(0, index=index, columns=columns)

        return df

    def get_unified_trajectory(self):
        """
        Calculate unified trajectory in per capita variables

        Returns
        -------
        Dataframe of unified per capita variables if calculation succeeds,
        else return -1 for TypeError and -2 for ValueError
        """

        L = self.dependent_vars['L']
        columns = ['k_c', 'k_d', 'l_c', 'l_d', 'g', 'c', 'r',
                   'n_c', 'i_c', 'r_c', 'r_d', 'w',
                   'W_c', 'W_d', 'Y_c', 'Y_d']
        var_expressions = [(self.independent_vars['K_c^c'] + self.independent_vars['K_c^d']) / L,
                           (self.independent_vars['K_d^c'] + self.independent_vars['K_d^d']) / L,
                           self.dependent_vars['Lc'] / L,
                           self.dependent_vars['Ld'] / L,
                           self.independent_vars['G'] / L,
                           self.independent_vars['C'] / L,
                           self.dependent_vars['R'] / L,
                           (self.independent_vars['x'] + 1.) / 2.,
                           (self.dependent_vars['rc'] * self.independent_vars['K_c^c']
                            + self.dependent_vars['rd'] * self.independent_vars['K_d^c'])
                           / (self.dependent_vars['rc'] * (self.independent_vars['K_c^c']
                                                           + self.independent_vars['K_c^d'])
                              + self.dependent_vars['rd'] * (self.independent_vars['K_d^c']
                                                             + self.independent_vars['K_d^d'])),
                           self.dependent_vars['rc'],
                           self.dependent_vars['rd'],
                           self.dependent_vars['w'],
                           self.dependent_vars['W_c'] / self.p_n,
                           self.dependent_vars['W_d'] / self.p_n,
                           self.p_b_c * self.independent_vars['C']**self.p_xi
                               *(self.independent_vars['K_c^c'] + self.independent_vars['K_c^d'])**self.p_kappa_c
                               *self.dependent_vars['Lc']**self.p_pi,
                           self.p_b_d * (self.independent_vars['K_d^c']
                              + self.independent_vars['K_d^d']) ** self.p_kappa_c
                           * self.dependent_vars['Ld'] ** self.p_pi
                           ]

        return self.calculate_unified_trajectory(columns=columns,
                                                 var_expressions=var_expressions)

    def get_event_rate_data(self):
        """
                Calculate unified trajectory in per capita variables

                Returns
                -------
                Dataframe of unified per capita variables if calculation succeeds,
                else return -1 for TypeError and -2 for ValueError
                """

        columns = ['W_tot', 'W_i_cd', 'W_i_dc', 'W_in_cd', 'W_in_dc', 'W_a', 'W_an', 'I_c', 'I_d', 'Pcd', 'Pdc', 'W_0']
        var_expressions = [1./self.p_tau,
                           self.dependent_vars['W_i_cd'],
                           self.dependent_vars['W_i_dc'],
                           self.dependent_vars['W_in_cd'],
                           self.dependent_vars['W_in_dc'],
                           self.dependent_vars['W_a'],
                           self.dependent_vars['W_an'],
                           self.dependent_vars['W_c'],
                           self.dependent_vars['W_d'],
                           self.dependent_vars['Pcd'],
                           self.dependent_vars['Pdc'],
                           self.dependent_vars['W_0']
                           ]

        df_out = self.calculate_unified_trajectory(columns=columns,
                                                 var_expressions=var_expressions)

        return(df_out)



if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt

    # investment_decisions:
    nopinions = [100, 100]

    # network:
    N = sum(nopinions)
    k = 10

    # building initial conditions
    p = float(k) / N
    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()
    investment_decisions = np.random.randint(low=0, high=2, size=N)

    clean_investment = np.ones(N) * 50. / float(N)
    dirty_investment = np.ones(N) * 50. / float(N)

    init_conditions = (adjacency_matrix, investment_decisions,
                       clean_investment, dirty_investment)

    m = IntegrateEquationsAggregate(*init_conditions, test=True)

    m.run(t_max=10)

    print('done runnin')

    fig, ax = plt.subplots(1)

    trj = m.get_event_rate_data()
    print(trj)
    trj.plot(ax=ax)
    plt.show()
