# Copyright (C) 2016-2018 by Jakob J. Kolb at Potsdam Institute for Climate
# Impact Research
#
# Contact: kolb@pik-potsdam.de
# License: GNU AGPL Version 3

import datetime
import os
import sys
import traceback
from contextlib import contextmanager
from itertools import chain
from random import shuffle

import networkx as nx
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.sparse.csgraph import connected_components
from scipy.stats import linregress

"""
From https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
"""

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()

    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")

    return fd

@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

"#################################################################"

class DivestmentCore:
    def __init__(self,
                 adjacency=None,
                 opinions=None,
                 investment_clean=None,
                 investment_dirty=None,
                 possible_cue_orders=None,
                 investment_decisions=None,
                 tau=0.8,
                 phi=.7,
                 eps=0.05,
                 L=1.,
                 G_0=3000,
                 C=1.,
                 b_c=1.,
                 b_d=1.5,
                 s=0.23,
                 d_c=0.06,
                 d_k=0.06,
                 b_r0=1.,
                 e=10,
                 xi=1. / 8.,
                 pi=1. / 2.,
                 kappa_c=1. / 2.,
                 kappa_d=1. / 2.,
                 mu=-2,
                 G=None,
                 R_depletion=True,
                 test=False,
                 learning=False,
                 campaign=False,
                 interaction=1,
                 verbosity=0,
                 **kwargs):
        """

        Parameters
        ----------
        adjacency: np.ndarray
            Acquaintance matrix between the households. Has to be symmetric
            unweighted and without self loops.
        possible_cue_orders: list
            A list of possible cue orders. Cue orders are integers that have
            to be implemented in the dict of
            possible cues with their respective callables in the init.
        opinions: list[int]
            A list, specifying for each household a cue order of the above
        investment_clean: list[float]
            Initial household endowments in the clean sector
        investment_dirty: list[float]
            Initial household endowments in the dirty sector
        investment_decisions: list[int]
            Initial investment decisions of households. Will be updated
            from their actual heuristic decision making during initialization
        tau: float
            Mean waiting time between household opinion updates
        phi: float
            Rewiring probability in the network adaptation process
        eps: float
            fraction of exploration events (noise) in the opinion formation
            process
        L: float
            Total labor (fixed)
        G_0: float
            Total initial resource stock
        C: float
            Total initial knowledge stock
        b_c: float
            Solow residual of the production function of the clean sector
        b_d: float
            Solow residual of the production function of the dirty sector
        s: float
            Savings rate of the households
        d_c: float
            Knowledge depreciation rate
        d_k: float
            Capital depreciation rate
        b_r0: float
            Resource cost factor
        e: float
            Resource efficiency in the dirty sector
        R_depletion: bool
            Switch to turn resource depreciation on or off
        test: bool
            switch for verbose output for debugging
        pi: float
            labor elasticity (equal in both sectors)
        kappa_c: float
            capital elasticity in the clean sector.
        kappa_d: float
            capital elasticity in the dirty sector.
        xi: float
            Elasticity of knowledge stock in the production process in the
            clean sector
        learning: bool
            Switch to toggle learning in the clean sector.
            If False, the knowledge stock is set to 1 and its dynamics are
            turned off.
        campaign: bool
            Switch to toggle separate treatment of zealots (campaigners)
            in the model, such that they do not immiatate
            other households decisions.
        interaction: int
            Switch for different imitation probabilities.
            if 0: tanh(Wi-Wj) interaction,
            if 1: interaction as in Traulsen, 2010 but with relative
            differences
            if 2: (Wi-Wj)/(Wi+Wj) interaction.
            if 3: random imitation e.g. p_cd = p_dc = .5
        t_trend: float
            length of running window average that chartes use to predict trends
        """

        if test:

            def verboseprint(*args):
                if self.verbosity > 2:
                    for stuff in args:
                        print(stuff)
        else:

            def verboseprint(*args):
                pass

        self.verboseprint = verboseprint

        if test:
            self.verbosity = 2
        else:
            self.verbosity = verbosity

        # Modes:
        #  1: only economy,
        #  2: economy + opinion formation + decision making,

        if possible_cue_orders is None:
            possible_cue_orders = [[0], [1]]

        # check, if heuristic decision making of imitation only
        self.heuristic_decision_making = False

        for p in possible_cue_orders:
            if p not in [[0], [1]]:
                self.heuristic_decision_making = True

        self.mode = 2

        # Agent Interactions:
        #  if 0: tanh(Wi-Wj) interaction,
        #  if 1: interaction as in Traulsen, 2010 but with relative differences
        #  if 2: (Wi-Wj)/(Wi+Wj) interaction.

        self.interaction = interaction

        # trajectory output time window

        if 'trj_output_window' in kwargs.keys():
            self.verboseprint('found trj_output_window')
            self.trj_output_window = kwargs['trj_output_window']
        else:
            self.trj_output_window = [0, np.float('inf')]

        # General Parameters

        # turn output for debugging on or off
        self.debug = test
        # toggle e_trajectory output
        self.e_trajectory_output = True
        self.m_trajectory_output = True
        self.switchlist_output = False
        # toggle whether to run full time or only until consensus
        self.run_full_time = True
        # toggle resource depletion
        self.R_depletion = R_depletion
        # toggle learning by doing
        self.learning = learning
        # toggle campaigning
        self.campaign = campaign
        # toggle imitation in avm
        self.imitation = True
        self.epsilon = np.finfo(dtype='float')

        # General Variables

        # System Time
        self.t = 0
        # Step counter for output
        self.steps = 0
        # eps == 0: 0 for no consensus, 1 consensus
        # eps>0 0 for no convergence, 1 for convergence at t_max
        self.consensus = False
        # variable to set if the model converged to some final state.
        self.converged = False
        # safes the system time at which consensus is reached
        self.convergence_time = float('NaN')
        # eps==0: -1 for no consensus, 1 for clean consensus,
        # 0 for dirty consensus, in between for fragmentation
        # eps>0: if converged: opinion state at time of convergence
        # if not converged: opinion state at t_max
        self.convergence_state = -1

        # dictionary of decision cues
        self.cues = {
            0: self.cue_0,
            1: self.cue_1,
            2: self.cue_2,
            3: self.cue_3,
            4: self.cue_4,
            5: self.cue_1
        }

        # list to save e_trajectory of output variables
        self.e_trajectory = []
        # list to save macroscopic quantities to compare with
        # moment closure / pair based proxy approach
        self.m_trajectory = []
        self.ag_trajectory = []
        # list of data for switching events
        self.switchlist = None
        # dictionary for final state
        self.final_state = {}

        # Household parameters

        # mean waiting time between social updates
        self.tau = tau
        # rewiring probability for adaptive voter model
        self.phi = phi
        # percentage of rewiring and imitation events that are noise
        self.eps = eps

        # number of households
        self.n = adjacency.shape[0]
        # birth rate for household members
        self.r_b = 0.
        # percentage of income saved
        self.s = s

        # Decision making variables:

        # number of steps that households memorize to estimate trend
        self.N_mem = 20
        # memory of r_c values
        self.r_cs = []
        # memory of r_d values
        self.r_ds = []
        # times of memories
        self.t_rs = []

        # Household variables

        # Individual

        # waiting times between rewiring events for each household
        self.waiting_times = \
            np.random.exponential(scale=self.tau, size=self.n)
        # adjacency matrix between households
        self.neighbors = adjacency
        # to select random investment_decisions,
        # all possible investment_decisions must be known
        self.possible_cue_orders = possible_cue_orders
        # investment_decisions as indices of possible_cue_orders
        self.opinions = np.array(opinions)
        # to keep track of the current ratio of investment_decisions
        self.clean_opinions = np.zeros((len(possible_cue_orders)))
        self.dirty_opinions = np.zeros((len(possible_cue_orders)))

        i, n = np.unique(self.opinions, return_counts=True)

        for j in range(len(possible_cue_orders)):
            if j not in list(i):
                n = np.append(n, [0])
                i = np.append(i, [j])
        self.opinion_state = [
            n[list(i).index(j)] if j in i else 0

            for j in range(len(self.possible_cue_orders))
        ]

        # to keep track of investment decisions.
        self.decision_state = 0.
        # investment decision vector, so far equal to investment_decisions

        if investment_decisions is None:
            if possible_cue_orders == [[0], [1]]:
                self.investment_decisions = np.array(opinions)
            else:
                self.investment_decisions = np.random.randint(0, 2, self.n)
        else:
            self.investment_decisions = investment_decisions

        # members of ALL household = population
        self.P = L

        # household investment in dirty capital

        if investment_dirty is None:
            self.investment_dirty = np.ones(self.n)
        else:
            self.investment_dirty = investment_dirty

        # household investment in clean capital

        if investment_clean is None:
            self.investment_clean = np.ones(self.n)
        else:
            self.investment_clean = investment_clean

        # household income (for social update)
        self.income = np.zeros(self.n)

        # Aggregated

        # total clean capital (supply)
        self.K_c = 0
        # total dirty capital (supply)
        self.K_d = 0
        self.K = self.K_c + self.K_d

        # Sector parameters

        # Capital depreciation rate
        self.d_k = d_k
        # Knowledge depreciation rate
        self.d_c = d_c
        # Resource harvest cost per unit (at full resource stock)
        self.b_r0 = b_r0
        # exponent of fossil resource in resource cost
        self.mu = mu

        # for Cobb Douglas economy
        # elasticities of labor and resource use are fixed
        # (pi = 2/5, rho = 3/4, epsilon = 5/4)
        # to be able to solve market clearing analytically

        # for Leontief dirty sector without profits,
        # capital and labor elasticities must be equal
        # in both sectors and satisfy pi + kappa = 1

        # solow residual for clean sector
        self.b_c = b_c
        # solow residual for dirty sector
        self.b_d = b_d

        # labor elasticity (equal in both sectors)
        self.pi = pi
        # elasticity of knowledge
        self.xi = xi
        # clean capital elasticity
        self.kappa_c = kappa_c
        # dirty capital elasticity
        self.kappa_d = kappa_d
        # fossil->energy->output conversion efficiency (Leontief)
        self.e = e

        # Sector variables

        self.P_c = L / 2.
        self.P_d = L / 2.

        self.K_c = 0.
        self.K_d = 0.

        # resource uptake in dirty sector
        self.R = 1.

        # knowledge stock in clean sector
        self.C = C if learning else 1

        self.Y_c = 0.
        self.Y_d = 0.

        # derived Sector variables

        self.w = 0.
        self.r_c = 0.
        self.r_c_dot = 0.
        self.r_d = 0.
        self.r_d_dot = 0.
        self.c_R = 0

        # Ecosystem parameters

        self.G_0 = G_0

        # Ecosystem variables
        if G is None:
            self.G = G_0
        else:
            self.G = G

        # calculate initial variables:

        dt = [self.t, self.t + 0.0001]
        x0 = np.fromiter(chain.from_iterable([
            list(self.investment_clean),
            list(self.investment_dirty), [self.P, self.G, self.C]
        ]),
                         dtype='float')
        with stdout_redirected():
            [x0, x1], self.db_out = odeint(self.economy_dot_leontief,
                                       x0,
                                       dt,
                                       full_output=True)

        self.investment_clean = x1[0:self.n]
        self.investment_dirty = x1[self.n:2 * self.n]
        self.P = x1[-3]
        self.G = x1[-2]
        self.C = x1[-1]

        if self.e_trajectory_output:
            self.init_economic_trajectory()

        if self.m_trajectory_output:
            self.init_mean_trajectory()
            self.init_aggregate_trajectory()

        if self.switchlist_output:
            self.init_switchlist()

        self.total_events = 0.

        self.imitation_cd_events = 0.
        self.imitation_dc_events = 0.
        self.noise_imitation_cd_events = 0.
        self.noise_imitation_dc_events = 0.

        self.adaptation_events = 0.
        self.noise_adaptation_events = 0.
        self.rate_data = None

    @staticmethod
    def cue_0(i):
        """
        evaluation of cue 0 for household i:
        Always decide for the dirty investment

        Parameters:
        -----------
        i : int
            index of the household that evaluates
            the cue
        Return:
        -------
        dec : int in [-1,0,1]
            decision output:
            -1 no decision, evaluate next cue
             0 decide for dirty investment
             1 decide for clean investment
        """

        dec = 0

        return dec

    @staticmethod
    def cue_1(i):
        """
        evaluation of cue 1 for household i:
        Always decide for the green investment

        Parameters:
        -----------
        i : int
            index of the household that evaluates
            the cue
        Return:
        -------
        dec : int in [-1,0,1]
            decision output:
            -1 no decision, evaluate next cue
             0 decide for dirty investment
             1 decide for clean investment
        """

        dec = 1

        return dec

    def cue_2(self, i):
        """
        evaluation of cue 2 for household i:
        Which rate of return is significantly higher?

        Parameters:
        -----------
        i : int
            index of the household that evaluates
            the cue
        Return:
        -------
        dec : int in [-1,0,1]
            decision output:
            -1 no decision, evaluate next cue
             0 decide for dirty investment
             1 decide for clean investment
        """
        dif = 1.

        if self.r_c > self.r_d * dif:
            dec = 1
        elif self.r_d > self.r_c * dif:
            dec = 0
        else:
            dec = -1

        return dec

    def cue_3(self, i):
        """
        evaluation of cue 3 for household i:
        do the trends of the rats differ?

        Parameters:
        -----------
        i : int
            index of the household that evaluates
            the cue
        Return:
        -------
        dec : int in [-1,0,1]
            decision output:
            -1 no decision, evaluate next cue
             0 decide for dirty investment
             1 decide for clean investment
        """

        if self.r_c_dot > self.r_d_dot * 1.1 and self.r_c > self.d_k:
            dec = 1
        elif self.r_d_dot > self.r_c_dot * 1.1 and self.r_d > self.d_k:
            dec = 0
        else:
            dec = -1

        return dec

    def cue_4(self, i):
        """
        evaluation of cue 4 for household i:
        What does the majority of the neighbors do?

        Parameters:
        -----------
        i : int
            index of the household that evaluates
            the cue
        Return:
        -------
        dec : int in [-1,0,1]
            decision output:
            -1 no decision, evaluate next cue
             0 decide for dirty investment
             1 decide for clean investment
        """
        threshold = 0.1

        neighbors = self.neighbors[:, i].nonzero()[0]
        n_ops = sum(self.investment_decisions[neighbors]) \
            / float(len(neighbors)) if len(neighbors) != 0 else .5

        if n_ops - threshold > 0.5:
            dec = 1
        elif n_ops + threshold < 0.5:
            dec = 0
        else:
            dec = -1

        return dec

    def set_parameters(self):
        """dummy function to mimic the interface of the
        approximation modules"""
        pass

    @staticmethod
    def progress(count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()

    def run(self, t_max=200.):
        """
        run model for t<t_max or until consensus is reached

        Parameter
        ---------
        t_max : float
            The maximum time the system is integrated [Default: 100]
            before run() exits. If the model reaches consensus, or is
            unable to find further update candidated, it ends immediately

        Return
        ------
        exit_status : int
            if exit_status == 1: consensus/convergence reached
            if exit_status == 0: no consensus/convergence reached at t=t_max
            if exit_status ==-1: no consensus, no update candidates found (BAD)
            if exit_status ==-2: economic model broken (BAD)
        """

        candidate = 0

        while self.t < t_max:

            if self.verbosity > 0:
                self.progress(self.t, t_max, 'abm running')

            self.verboseprint(self.t, t_max)

            # 1 find update candidate and respective update time
            (candidate, neighbor, neighbors,
             update_time) = self.find_update_candidates()

            # 2 integrate economic model until t=update_time:
            # don't make steps too large. The integrator handles that badly..

            if update_time - self.t < 1.:
                self.update_economy(update_time)
            else:
                while True:
                    inter_update_time = (self.t +
                                         1. if not self.t + 1. > update_time
                                         else update_time)
                    self.update_economy(inter_update_time)

                    if inter_update_time >= update_time:
                        break

            # 3 update opinion formation in case,
            # update candidate was found:

            if candidate >= 0:
                self.update_opinion_formation(candidate, neighbor, neighbors)

            # 4 update investment decision making:
            self.update_decision_making()

            # 5 check for 2/3 majority for clean investment
            self.detect_convergence(self.investment_decisions)

            if not self.run_full_time and self.converged:
                break

        # save final state to dictionary
        self.final_state = {
            'adjacency': self.neighbors,
            'opinions': self.opinions,
            'investment_decisions': self.investment_decisions,
            'investment_clean': self.investment_clean,
            'investment_dirty': self.investment_dirty,
            'possible_cue_orders': self.possible_cue_orders,
            'tau': self.tau,
            'phi': self.phi,
            'eps': self.eps,
            'L': self.P,
            'r_b': self.r_b,
            'b_c': self.b_c,
            'b_d': self.b_d,
            's': self.s,
            'd_c': self.d_c,
            'b_r0': self.b_r0,
            'e': self.e,
            'G_0': self.G,
            'C': self.C,
            'd_k': self.d_k,
            'xi': self.xi,
            'learning': self.learning,
            'campaign': self.campaign,
            'test': self.debug,
            'R_depletion': False
        }

        if self.converged:
            return 1  # good - consensus reached
        elif not self.converged and self.R_depletion:
            self.convergence_state = float('nan')
            self.convergence_time = self.t

            return 0  # no consensus found during run time
        elif candidate == -2:
            return -1  # bad run - opinion formation broken
        elif np.isnan(self.G):
            return -2  # bad run - economy broken
        else:
            return -3  # very bad run. Investigations needed

    def b_rf(self, resource):
        """
        Calculates the dependence of resource harvest cost on
        remaining resource stock starts at b_r0 and
        increases with decreasing stock if stock is depleted,
        costs are infinite

        Parameter
        ---------
        resource : float
            The quantity of resource remaining in Stock

        Return
        ------
        b_r     : float
            The resource extraction efficiency according to the
            current resource stock
        """

        if resource > 0:
            b_r = self.b_r0 * (resource/self.G_0)**self.mu
        else:
            b_r = float('inf')

        return b_r

    def economy_dot_leontief(self, x0, t):
        """
        economic model assuming Cobb-Douglas production
        for the clean sector and Leontief/Cobb-Douglas
        production for the dirty sector:

            Y_c = b_c P_c^pi_c K_c^kappa_c,
            Y_d = min(b_d P_d^pi_d K_d^kappa_d, e R).

        and linear resource extraction costs:

            c_R = b_R R

        where b_R depends on the remaining resource stock.
        It is also assumed that there is no profits e.g.

            Y_c - w P_c - r_c K_c = 0,
            Y_d - w P_d - r_d K_d - c_R = 0,

        and that labor elasticities are equal, e.g.

            pi_c = pi_d.

        Parameters:
        -----------

        x0  : list[float]
            state vector of the system of length
            2N + 2. First 2N entries are
            clean household investments [0:n] and
            dirty household investments [n:2N].
            Second last entry is total population
            The last entry is the remaining fossil
            reserves.

        t   : float
            the system time.

        Returns:
        --------
        x1  : list[floats]
            updated state vector of the system of length
            3N + 1. First 3N entries are numbers
            of household members (depreciated) [0:n]
            clean household investments [n:2N] and
            dirty household investments [2N:3N].
            The last entry is the remaining fossil
            reserves.
        """

        investment_clean = np.where(x0[0:self.n] > 0, x0[0:self.n],
                                    np.full(self.n, self.epsilon.eps))
        investment_dirty = np.where(x0[self.n:2 * self.n] > 0,
                                    x0[self.n:2 * self.n],
                                    np.full(self.n, self.epsilon.eps))
        P = x0[-3]
        G = x0[-2]
        C = x0[-1]

        K_c = sum(investment_clean)
        K_d = sum(investment_dirty)

        try:
            assert K_c >= 0, 'negative clean capital'
            assert K_d >= 0, 'negative dirty capital'
            assert G >= 0, 'negative resource'
            assert C >= 0, 'negative knowledge'
        except AssertionError:
            if self.debug is True:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)  # Fixed format
                tb_info = traceback.extract_tb(tb)
                filename, line, func, text = tb_info[-1]
                print(
                    f'An error occurred on line {line} in statement {text} with'
                )
                print(f'K_c = {K_c}, K_d = {K_d}, G = {G}, C = {C}')
                print('the trajectory tail:')
                trj = self.get_economic_trajectory()
                print(trj.tail(10))
                print('and last stats from odeint')
                print(self.db_out)
                print('model configuration is:')
                print(dir(self))

            if G < 0:
                G = 0

            if C < 0:
                C = 0

        b_R = self.b_rf(G)

        X_c = (self.b_c * C**self.xi * K_c**self.kappa_c)**(1. /
                                                            (1. - self.pi))
        X_d = (self.b_d * K_d**self.kappa_d)**(1. / (1. - self.pi))
        X_R = (1. - b_R / self.e) ** (1. / (1. - self.pi)) \
            if 1. > b_R / self.e else float('NaN')

        P_c = P * X_c / (X_c + X_d * X_R)
        P_d = P * X_d * X_R / (X_c + X_d * X_R)
        R = 1. / self.e * self.b_d * K_d**self.kappa_d * P_d**self.pi

        self.w = self.pi * P**(self.pi - 1) * (X_c + X_d * X_R)**(1. - self.pi)
        self.r_c = self.kappa_c / \
            K_c * X_c * P ** self.pi * (X_c + X_d * X_R) ** (- self.pi)
        self.r_d = self.kappa_d / \
            K_d * X_d * X_R * P ** self.pi * (X_c + X_d * X_R) ** (- self.pi)

        # check if dirty sector is profitable (P_d > 0).
        # if not, shut it down.

        if P_d < 0 or np.isnan(X_R):
            P_d = 0
            P_c = P
            R = 0
            self.w = self.b_c * C ** self.xi * K_c ** self.kappa_c\
                * self.pi * P ** (self.pi - 1.)
            self.r_c = self.b_c * C ** self.xi * self.kappa_c\
                * K_c ** (self.kappa_c - 1.) * P ** self.pi
            self.r_d = 0

        self.R = R
        self.K_c = K_c
        self.K_d = K_d
        self.P = P
        self.P_c = P_c
        self.P_d = P_d
        self.c_R = b_R * R

        self.income = (self.r_c * self.investment_clean +
                       self.r_d * self.investment_dirty + self.w * P / self.n)
        try:
            assert all([x > 0 for x in self.income])

        except AssertionError:
            print(f'after time t = {t}')
            print(f'tau = {self.tau}, phi = {self.phi}, b_d = {self.b_d}')
            print('income is negative, X_R: {}, X_d: {}, X_c: {}, \n '
                  'K_d: {}, K_c: {} , G = {}, C = {} \n '
                  'r_c = {}, r_d = {}, w = {}, R = {} \n '
                  'investment decisions: \n {} \n '
                  'income: \n {} \n'
                  'clean investment: \n {} \n'
                  'dirty investment: \n {}'.format(
                      X_R, X_d, X_c, K_d, K_c, G, C, self.r_c, self.r_d,
                      self.w, self.R, self.investment_decisions, self.income,
                      self.investment_clean, self.investment_dirty))

            exit(-1)

        G_dot = -R if self.R_depletion else 0.0
        P_dot = 0
        C_dot = self.b_c * C ** self.xi * P_c ** self.pi \
            * K_c ** self.kappa_c - C * self.d_c if self.learning else 0.
        investment_clean_dot = \
            self.investment_decisions \
            * self.s * self.income - self.investment_clean * self.d_k
        investment_dirty_dot = \
            np.logical_not(self.investment_decisions) \
            * self.s * self.income - self.investment_dirty * self.d_k

        x1 = np.fromiter(chain.from_iterable([
            list(investment_clean_dot),
            list(investment_dirty_dot), [P_dot, G_dot, C_dot]
        ]),
                         dtype='float')

        return x1

    def update_economy(self, update_time):
        """
        Integrates the economic equations of the
        model until the system time equals the update time.

        Also keeps track of the capital return rates and estimates
        the time derivatives of capital return rates trough linear
        regression.

        Finally, appends the current system state to the system e_trajectory.

        Parameters:
        -----------
        self : object
            instance of the model class
        update_time : float
            time until which system is integrated
        """

        dt = [self.t, update_time]
        x0 = np.fromiter(chain.from_iterable([
            list(self.investment_clean),
            list(self.investment_dirty), [self.P, self.G, self.C]
        ]),
                         dtype='float')

        # integrate the system unless it crashes.

        if not np.isnan(self.R):
            with stdout_redirected():
                [x0, x1] = odeint(self.economy_dot_leontief, x0, dt, mxhnil=1)
        else:
            x1 = x0

        self.investment_clean = np.where(x1[0:self.n] > 0, x1[0:self.n],
                                         np.zeros(self.n))
        self.investment_dirty = np.where(x1[self.n:2 * self.n] > 0,
                                         x1[self.n:2 * self.n],
                                         np.zeros(self.n))
        self.P = x1[-3]
        self.G = x1[-2]
        self.C = x1[-1]

        # memorize return rates for trend estimation.
        # this is only necessary for heuristic decision making.
        # for imitation only, this can be switched off.

        if self.heuristic_decision_making:
            self.r_cs.append(self.r_c)
            self.r_ds.append(self.r_d)
            self.t_rs.append(self.t)

            if len(self.r_cs) > self.N_mem:
                self.r_cs.pop(0)
                self.r_ds.pop(0)
                self.t_rs.pop(0)

            # estimate trends in capital returns
            self.r_c_dot = linregress(self.t_rs, self.r_cs)[0]
            self.r_d_dot = linregress(self.t_rs, self.r_ds)[0]

            if np.isnan(self.r_c_dot):
                self.r_c_dot = 0

            if np.isnan(self.r_d_dot):
                self.r_d_dot = 0

        self.t = update_time
        self.steps += 1

        # calculate market shares:
        self.Y_c = self.b_c * self.C ** self.xi \
            * self.K_c ** self.kappa_c * self.P_c ** self.pi
        self.Y_d = self.b_d * self.K_d**self.kappa_d * self.P_d**self.pi

        # output economic data if t is in time window

        if self.trj_output_window[
                0] - self.tau < self.t < self.trj_output_window[1] + self.tau:

            if self.e_trajectory_output:
                self.update_economic_trajectory()

            if self.m_trajectory_output:
                self.update_mean_trajectory()
                self.update_aggregate_trajectory()
            self.update_event_rate_data()

    def find_update_candidates(self):

        i, n = np.unique(self.opinions, return_counts=True)
        self.opinion_state = [
            n[list(i).index(j)] if j in i else 0

            for j in range(len(self.possible_cue_orders))
        ]
        i = 0
        i_max = 1000 * self.n
        candidate = 0
        neighbor = self.n
        neighbors = []
        update_time = self.t

        while i < i_max:

            # find household with min waiting time
            candidate = self.waiting_times.argmin()

            # remember update_time and increase waiting time of household
            update_time = self.waiting_times[candidate]
            self.waiting_times[candidate] += \
                np.random.exponential(scale=self.tau)

            # count household activation event:
            self.total_events += 1. / self.n

            # determine if event is a noise event.
            rdn = np.random.uniform()

            # with prob. eps*(1-phi) it is a noise imitation event:
            # and the household choses its strategy uniformly from
            # the available strategies.

            if rdn < self.eps * (1 - self.phi) and self.imitation:
                # save old opinion
                old_opinion = self.opinions[candidate]
                # determine new opinion
                new_opinion = np.random.randint(len(self.possible_cue_orders))
                self.opinions[candidate] = new_opinion
                # if required save switching data
                # if old_opinion != new_opinion and self.switchlist_output:
                #     self.save_switch(candidate, old_opinion)
                # and count event to determine rates - if it changes macro
                # properties.

                if old_opinion != new_opinion:
                    if new_opinion == 1:
                        self.noise_imitation_dc_events += 1. / self.n
                    else:
                        self.noise_imitation_cd_events += 1. / self.n
                candidate = -1

                break

            # with prob. p = eps*phi it is a noise adaptation event
            # and the household rewires to a random new neighbor.
            elif rdn > 1. - self.eps * self.phi and len(neighbors) > 0:
                unconnected = np.zeros(self.n, dtype=int)

                for i in range(self.n):
                    if i not in neighbors:
                        unconnected[i] = 1
                unconnected = unconnected.nonzero()[0]

                if len(unconnected) > 0:
                    old_neighbor = np.random.choice(neighbors)
                    new_neighbor = np.random.choice(unconnected)
                    self.neighbors[candidate, old_neighbor] = \
                        self.neighbors[old_neighbor, candidate] = 0
                    self.neighbors[candidate, new_neighbor] = \
                        self.neighbors[new_neighbor, candidate] = 1
                    # count event, if it changed macro properties
                    old_opinion = self.opinions[old_neighbor]
                    new_opinion = self.opinions[new_neighbor]

                    if new_opinion != old_opinion:
                        self.noise_adaptation_events += 1. / self.n
                candidate = -1

                break

            # load neighborhood of household i
            neighbors = self.neighbors[:, candidate].nonzero()[0]

            # if candidate has neighbors, chose one at random.

            if len(neighbors) > 0:
                neighbor = np.random.choice(neighbors)

                # check if preferences of candidate and random
                # neighbor differ

                if self.opinions[candidate] == self.opinions[neighbor]:
                    # if candidate and neighbor have same preferences, they
                    # not suitable for update. (RETRY)
                    neighbor = self.n

            if neighbor < self.n:
                # update candidate found (GOD)

                break
            else:
                i += 1

                if i % self.n == 0:
                    if self.detect_consensus_state(self.opinions):
                        # no update candidate found because of
                        # consensus state (GOD)
                        candidate = -1

                        break

            if i >= i_max:
                # no update candidate and no consensus found (BAD)
                candidate = -2

        return candidate, neighbor, neighbors, update_time

    def update_opinion_formation(self, candidate, neighbor, neighbors):

        same_unconnected = np.zeros(self.n, dtype=int)
        opinion = self.opinions

        # adapt or rewire?

        if (self.phi == 1
                or (self.phi != 1 and np.random.uniform() < self.phi)):
            # rewire:

            for i in range(self.n):
                # find potential new neighbors:
                # campaigners rewire to everybody

                if (self.campaign is True and opinion[candidate] == len(
                        self.possible_cue_orders)):
                    same_unconnected[i] = 1

                # everybody else rewires to people with same opinion
                else:
                    if opinion[i] == opinion[
                            candidate] and i not in neighbors and i != candidate:
                        same_unconnected[i] = 1
            same_unconnected = same_unconnected.nonzero()[0]

            if len(same_unconnected) > 0:
                # if there are potential new neighbors, connect to one of them:
                # 1) select neighbor
                new_neighbor = np.random.choice(same_unconnected)
                # 2) update network
                self.neighbors[candidate, neighbor] = \
                    self.neighbors[neighbor, candidate] = 0
                self.neighbors[candidate, new_neighbor] = \
                    self.neighbors[new_neighbor, candidate] = 1
                # 3) count event (and normalize to get the rate)
                self.adaptation_events += 1. / self.n
        else:
            # adapt:
            # compare fitness
            Wi = self.fitness(candidate)
            Wj = self.fitness(neighbor)

            if self.interaction == 0:
                p_imitate = .5 * (np.tanh(Wj - Wi) + 1)
            elif self.interaction == 1:
                p_imitate = 1. / (1 + np.exp(-8. * (Wj - Wi) / (Wj + Wi)))
            elif self.interaction == 2:
                p_imitate = .5 * ((Wj - Wi) / (Wj + Wi) + 1)
            elif self.interaction == 3:
                p_imitate = .5
            else:
                raise ValueError(
                    'interaction not defined, must be in [0, 1, 2] but is {}'.
                    format(self.interaction))
            # and determine wheter imitation happens:

            if ((self.campaign is False or opinion[candidate] !=
                 (len(self.possible_cue_orders) - 1))
                    and (np.random.uniform() < p_imitate) and self.imitation):
                imitation_happened = True
                # copy opinion
                self.opinions[candidate] = self.opinions[neighbor]
                # count event:

                if self.opinions[candidate] == 1:
                    self.imitation_dc_events += 1. / self.n
                else:
                    self.imitation_cd_events += 1. / self.n
            else:
                imitation_happened = False

            # and if required, save imitation data.

            if self.switchlist_output:

                direction = self.opinions[neighbor]

                self.save_switch(household_index=candidate,
                                 direction=direction,
                                 probability=p_imitate,
                                 income_1=Wi,
                                 income_2=Wj,
                                 event=imitation_happened)

        return 0

    def update_decision_making(self):
        """
        Updates the investment decision for all households depending on their
        cue orders (opinion) and the state of the economy
        """

        self.dirty_opinions = np.zeros((len(self.possible_cue_orders)))
        self.clean_opinions = np.zeros((len(self.possible_cue_orders)))

        for i in range(self.n):
            for cue in self.possible_cue_orders[self.opinions[i]]:
                decision = self.cues[cue](i)

                if decision != -1:
                    self.investment_decisions[i] = decision

                    break

            if decision == -1:
                self.investment_decisions[i] = np.random.randint(2)
                decision = self.investment_decisions[i]

            if decision == 0:
                self.dirty_opinions[self.opinions[i]] += 1. / self.n
            elif decision == 1:
                self.clean_opinions[self.opinions[i]] += 1. / self.n

        self.decision_state = np.mean(self.investment_decisions)

        return 0

    def detect_consensus_state(self, opinions):
        # check if network is split in components with
        # same investment_decisions/preferences
        # returns 1 if consensus state is detected,
        # returns 0 if NO consensus state is detected.

        cc = connected_components(self.neighbors, directed=False)[1]
        self.consensus = all(
            len(np.unique(opinions[c])) == 1

            for c in ((cc == i).nonzero()[0] for i in np.unique(cc)))

        if self.eps == 0:
            if self.consensus and self.convergence_state == -1:
                self.convergence_state = np.mean(opinions)
                self.convergence_time = self.t
                self.converged = True

        return self.converged

    def detect_convergence(self, opinions):
        """
        check, if the system converged
        to some attractor.
        If the system converged, set convergence time.

        Parameters:
        -----------
        investment_decisions: [int]
            list of investment_decisions.
            opinion==1 : clean
            opinion==0 : dirty

        Return:
        -------
        dist : float
            distance from attractor
            1: far
            0: reached
        """

        state = float(sum(opinions)) / float(len(opinions))

        attractor = 2. / 3.
        dist = attractor - state
        self.verboseprint(dist, self.t, self.convergence_state)
        alpha = (self.b_r0 / self.e)**(1. / 2.)

        if self.eps > 0 and dist < 0. and np.isnan(self.convergence_time):
            self.convergence_state = \
                (self.G - alpha * self.G_0) / (self.G_0 * (1. - alpha))
            self.convergence_time = self.t
            self.converged = True

    def fitness(self, agent):
        return self.income[agent]

    def init_economic_trajectory(self):
        element = list(
            chain.from_iterable([[
                'time', 'wage', 'r_c', 'r_d', 'r_c_dot', 'r_d_dot', 'K_c',
                'K_d', 'P_c', 'P_d', 'L', 'G', 'R', 'C', 'Y_c', 'Y_d',
                'P_c_cost', 'P_d_cost', 'K_c_cost', 'K_d_cost', 'c_R',
                'consensus', 'decision state', 'G_alpha', 'i_c'
            ], [str(x) for x in self.possible_cue_orders
                ], ['c' + str(x) for x in self.possible_cue_orders
                    ], ['d' + str(x) for x in self.possible_cue_orders]]))
        self.e_trajectory.append(element)

        dt = [self.t, self.t]
        x0 = np.fromiter(chain.from_iterable([
            list(self.investment_clean),
            list(self.investment_dirty), [self.P, self.G, self.C]
        ]),
                         dtype='float')
        with stdout_redirected():
            [x0, x1] = odeint(self.economy_dot_leontief, x0, dt)

        self.investment_clean = x1[0:self.n]
        self.investment_dirty = x1[self.n:2 * self.n]
        self.P = x1[-3]
        self.G = x1[-2]
        self.C = x1[-1]

        if self.trj_output_window[
                0] - self.tau < self.t < self.trj_output_window[1] + self.tau:
            self.update_economic_trajectory()

    def update_economic_trajectory(self):
        alpha = (self.b_r0 / self.e)**(1. / 2.)
        element = list(
            chain.from_iterable([[
                self.t, self.w, self.r_c, self.r_d, self.r_c_dot, self.r_d_dot,
                self.K_c, self.K_d, self.P_c, self.P_d, self.P, self.G, self.R,
                self.C, self.Y_c, self.Y_d, self.P_c * self.w,
                self.P_d * self.w, self.K_c * self.r_c, self.K_d * self.r_d,
                self.c_R, self.converged, self.decision_state,
                (self.G - alpha * self.G_0) / (self.G_0 * (1. - alpha)),
                sum(self.income * self.investment_decisions) /
                sum(self.income) if sum(self.income) > 0 else 0
            ], self.opinion_state, self.clean_opinions, self.dirty_opinions]))
        self.e_trajectory.append(element)

    def get_economic_trajectory(self):
        # make up DataFrame from micro data
        columns = self.e_trajectory[0]
        df = pd.DataFrame(data=self.e_trajectory[1:], columns=columns)
        df = df.set_index('time')

        return df

    def init_mean_trajectory(self):
        """
        This function initializes the e_trajectory for the output of the
        macroscopic quantitites as computed via moment closure and
        pair based proxy.
        :return: None
        """
        element = [
            'time', 'x', 'y', 'z', 'mu_c^c', 'mu_d^c', 'mu_c^d', 'mu_d^d', 'c',
            'g', 'N_c over N', '[cc] over M', '[cd] over M'
        ]
        self.m_trajectory.append(element)

        if self.trj_output_window[
                0] - self.tau < self.t < self.trj_output_window[1] + self.tau:
            self.update_mean_trajectory()

    def update_mean_trajectory(self):
        """
        This function calculates the macroscopic variables that are
        the dynamic variables in the macroscopic approximation and saves
        them in the e_trajectory list.
        :return: None
        """

        def cl(adj, x, y):
            """
            calculate number of links between like links in x and y
            :param adj: adjacency matrix
            :param x: node vector
            :param y: node vector
            :return: number of like links
            """
            assert len(x) == len(y)

            return float(np.dot(x, np.dot(adj, y)))

        adj = self.neighbors
        c = self.investment_decisions
        d = -self.investment_decisions + 1

        n = self.n
        k = float(sum(sum(self.neighbors))) / 2

        nc = sum(self.investment_decisions)
        nd = sum(-self.investment_decisions + 1)

        cc = cl(adj, c, c) / 2
        cd = cl(adj, c, d)
        dd = cl(adj, d, d) / 2

        x = float(nc - nd) / n
        y = float(cc - dd) / k
        z = float(cd) / k

        if nc > 0:
            mucc = sum(self.investment_decisions * self.investment_clean) / nc
            mudc = sum(self.investment_decisions * self.investment_dirty) / nc
        else:
            mucc = mudc = 0

        if nd > 0:
            mucd = sum(
                (1 - self.investment_decisions) * self.investment_clean) / nd
            mudd = sum(
                (1 - self.investment_decisions) * self.investment_dirty) / nd
        else:
            mucd = mudd = 0

        entry = [
            self.t, x, y, z, mucc, mudc, mucd, mudd, self.C / n, self.G / n,
            nc / n, cc / k, cd / k
        ]
        self.m_trajectory.append(entry)

    def get_mean_trajectory(self):
        # make up Dataframe from macro data:
        columns = self.m_trajectory[0]
        df = pd.DataFrame(self.m_trajectory[1:], columns=columns)
        df = df.set_index('time')

        return df

    def init_aggregate_trajectory(self):
        """
        This function initializes the e_trajectory for the output of the
        macroscopic quantitites as computed via moment closure and
        pair based proxy.
        :return: None
        """
        element = [
            'time', 'x', 'y', 'z', 'K_c^c', 'K_d^c', 'K_c^d', 'K_d^d', 'C',
            'G', 'w', 'r_c', 'r_d', 'W_c', 'W_d', 'N_c over N', '[cc] over M',
            '[cd] over M'
        ]
        self.ag_trajectory.append(element)

        if (self.trj_output_window[0] - self.tau < self.t <
                self.trj_output_window[1] + self.tau):
            self.update_aggregate_trajectory()

    def update_aggregate_trajectory(self):
        """
        This function calculates the macroscopic variables that are
        the dynamic variables in the macroscopic approximation and saves
        them in the e_trajectory list.
        :return: None
        """

        def cl(adj, x, y):
            """
            calculate number of links between x and y
            :param adj: adjacency matrix
            :param x: node vector
            :param y: node vector
            :return: number of like links
            """
            assert len(x) == len(y)

            return float(np.dot(x, np.dot(adj, y)))

        adj = self.neighbors
        c = self.investment_decisions
        d = -self.investment_decisions + 1

        # number of nodes aka. N
        n = self.n
        # number of edges, aka. M
        k = float(sum(sum(adj))) / 2

        nc = sum(c)
        nd = sum(d)

        cc = cl(adj, c, c) / 2
        cd = cl(adj, c, d)
        dd = cl(adj, d, d) / 2

        x = float(nc - nd) / n
        y = float(cc - dd) / k
        z = float(cd) / k

        if nc > 0:
            Kcc = sum(self.investment_decisions * self.investment_clean)
            Kdc = sum(self.investment_decisions * self.investment_dirty)
            Wc = Kcc / nc * self.r_c + Kdc / nc * self.r_d
        else:
            Kcc = Kdc = 0
            Wc = 0.

        if nd > 0:
            Kcd = sum((1 - self.investment_decisions) * self.investment_clean)
            Kdd = sum((1 - self.investment_decisions) * self.investment_dirty)
            Wd = Kcd / nd * self.r_c + Kdd / nd * self.r_d
        else:
            Kcd = Kdd = 0
            Wd = 0.

        entry = [
            self.t, x, y, z, Kcc, Kdc, Kcd, Kdd, self.C, self.G, self.w,
            self.r_c, self.r_d, Wc, Wd, nc / n, cc / k, cd / k
        ]
        self.ag_trajectory.append(entry)

    def get_aggregate_trajectory(self):
        # make up Dataframe from macro data:
        columns = self.ag_trajectory[0]
        self.verboseprint(columns)
        df = pd.DataFrame(self.ag_trajectory[1:], columns=columns)
        df = df.set_index('time')

        return df

    def init_switchlist(self):
        """Initializes the switchlist by naming the columns"""
        self.switchlist = [[
            'time', '$K^{(c)}$', '$K^{(d)}$', 'Direction',
            'Income of active agent', 'Income of neighbor', 'Imitation',
            'Probability'
        ]]

    def save_switch(self, household_index, direction, probability, income_1,
                    income_2, event):
        """
        Adds an entry to the switchlist.

        Parameters
        ----------
        household_index : int
            the index of the household that [might have] switched its opinion
        direction : int
            the direction that it switched in
            0 neighbors opinion is dirty
            1 neighbors opinion is clean
        probability: float
            probability of imitation
        income_1: float
            income of active agent
        income_2: float
            income of neighbor
        event: bool
            indicates if imitation actually happened
        """

        if self.switchlist is None:
            self.init_switchlist()

        self.switchlist.append([
            float(self.t),
            float(self.investment_clean[household_index]),
            float(self.investment_dirty[household_index]),
            int(direction),
            float(income_1),
            float(income_2),
            bool(event),
            float(probability)
        ])

    def get_switch_list(self):
        columns = self.switchlist[0]
        df = pd.DataFrame(self.switchlist[1:],
                          columns=columns).convert_objects()
        df = df.set_index('time')

        return df

    def update_event_rate_data(self):
        """write rate data to trajectory"""

        if self.rate_data is None:
            self.rate_data = [[
                'time', 'E_tot', 'E_i_cd', 'E_i_dc', 'E_in_cd', 'E_in_dc',
                'E_a', 'E_an'
            ]]
            self.rate_data.append([0, 0, 0, 0, 0])
        else:
            self.rate_data.append([
                self.t, self.total_events, self.imitation_cd_events,
                self.imitation_dc_events, self.noise_imitation_cd_events,
                self.noise_imitation_dc_events, self.adaptation_events,
                self.noise_adaptation_events
            ])

    def get_event_rate_data(self):
        """return dataframe with event rate data and time as index values"""
        columns = self.rate_data[0]
        df = pd.DataFrame(self.rate_data[1:], columns=columns)

        return df.set_index('time')

    def get_unified_trajectory(self):
        """
        Calculates unified trajectory in per capita variables

        Returns
        -------
        Dataframe of unified per capita variables
        """

        if self.e_trajectory is False:
            print('calculating the economic trajectory is a prerequisite.'
                  'please rerun the model with e_trajectory set to true.')

            return -1

        if self.m_trajectory is False:
            print('needs mean trajectory to be enabled.')

            return -1
        mdf = self.get_mean_trajectory()

        columns = [
            'k_c', 'k_d', 'l_c', 'l_d', 'g', 'c', 'r', 'n_c', 'i_c', 'r_c',
            'r_d', 'w', 'W_c', 'W_d'
        ]
        edf = self.get_economic_trajectory()
        df = pd.DataFrame(index=edf.index, columns=columns)
        df['k_c'] = edf['K_c'] / self.P
        df['k_d'] = edf['K_d'] / self.P
        df['l_c'] = edf['P_c'] / self.P
        df['l_d'] = edf['P_d'] / self.P
        df['g'] = edf['G'] / self.P
        df['c'] = edf['C'] / self.P
        df['r'] = edf['R'] / self.P
        df['n_c'] = edf['[1]'] / self.n
        df['i_c'] = edf['i_c']
        df['r_c'] = edf['r_c']
        df['r_d'] = edf['r_d']
        df['w'] = edf['wage']
        df['W_c'] = mdf['mu_c^c'] * edf['r_c'] + mdf['mu_d^c'] * edf['r_d']
        df['W_d'] = mdf['mu_c^d'] * edf['r_c'] + mdf['mu_d^d'] * edf['r_d']

        return df


if __name__ == '__main__':
    """
    Perform test run and plot some output to check
    functionality
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as mp

    output_location = 'test_output/' \
        + datetime.datetime.now().strftime("%d_%m_%H-%M-%Ss") + '_output'

    # Initial conditions:
    FFH = False

    if FFH:
        nopinions = [10, 10, 10, 10, 10, 10, 10, 10]
        possible_opinions = [
            [2, 3],  # short term investor
            [3, 2],  # long term investor
            [4, 2],  # short term herder
            [4, 3],  # trending herder
            [4, 1],  # green conformer
            [4, 0],  # dirty conformer
            [1],  # gutmensch
            [0]
        ]  # redneck
        input_parameters = {
            'tau': 1,
            'eps': 0.05,
            'b_d': 1.2,
            'b_c': 1.,
            'phi': 0.8,
            'e': 100,
            'G_0': 1500,
            'b_r0': 0.1**2 * 100,
            'possible_cue_orders': possible_opinions,
            'C': 1,
            'xi': 1. / 8.,
            'beta': 0.06,
            'campaign': False,
            'learning': True
        }

    if not FFH:
        # investment_decisions:
        nopinions = [10, 10]
        possible_cue_orders = [[0], [1]]

        # Parameters:

        input_parameters = {
            'tau': 1,
            'eps': 0.01,
            'b_d': 1.2,
            'b_c': 1.,
            'phi': 0.8,
            'e': 100,
            'G_0': 800,
            'b_r0': 0.1**2 * 100,
            'possible_cue_orders': possible_cue_orders,
            'C': 1,
            'xi': 1. / 8.,
            'beta': 0.06,
            'campaign': False,
            'learning': True
        }

    cops = ['c' + str(x) for x in possible_cue_orders]
    dops = ['d' + str(x) for x in possible_cue_orders]
    colors = [np.random.rand(3, 1) for x in possible_cue_orders]
    colors = colors + colors

    opinions = []

    for i, n in enumerate(nopinions):
        opinions.append(np.full((n), i, dtype='I'))
    opinions = [item for sublist in opinions for item in sublist]
    shuffle(opinions)

    # network:
    N = sum(nopinions)
    p = 10. / N

    while True:
        net = nx.erdos_renyi_graph(N, p)

        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    (mucc, mucd, mudc, mudd) = (1, 1, 1, 1)

    op = np.array(opinions)

    clean_investment = mucc * (op) + mudc * (1 - op)
    dirty_investment = mucd * (op) + mudd * (1 - op)

    init_conditions = (adjacency_matrix, opinions, clean_investment,
                       dirty_investment)

    # Initialize Model

    model = DivestmentCore(*init_conditions, **input_parameters)

    print('heuristic decision making', model.heuristic_decision_making)

    # Turn off economic trajectory
    model.e_trajectory_output = True

    # Turn on debugging
    model.debug = True

    # Run Model
    model.R_depletion = False
    model.run(t_max=100)
    model.R_depletion = True
    model.run(t_max=600)

    # Print some output

    print(connected_components(model.neighbors, directed=False))
    print('investment decisions:')
    print(model.investment_decisions)
    print('consensus reached?', model.converged)
    print(model.convergence_state)
    print('finish time', model.t)
    print('steps computed', model.steps)

    model.get_unified_trajectory()

    colors = [c for c in 'gk']

    df = model.get_unified_trajectory()
    print(df.columns)
    columns = [
        'k_c', 'k_d', 'l_c', 'l_d', 'g', 'c', 'r', 'n_c', 'i_c', 'r_c', 'r_d',
        'w'
    ]
    fig = mp.figure()
    ax1 = fig.add_subplot(221)
    df[['r_c', 'r_d']].plot(ax=ax1, style=colors)
    ax1b = ax1.twinx()
    df[['w']].plot(ax=ax1b)

    ax2 = fig.add_subplot(223)
    df[['n_c']].plot(ax=ax2)

    ax3 = fig.add_subplot(224)
    df[['k_c', 'k_d']].plot(ax=ax3, style=colors)

    ax4 = fig.add_subplot(222)
    df[['g']].plot(ax=ax4, style=colors[1])
    ax5 = ax4.twinx()
    df[['c']].plot(ax=ax5, style=colors[0])

    fig.tight_layout()
    fig.savefig('example_trajectory.png')

    rates = model.get_event_rate_data()

    print(rates)

    figs, axes = mp.subplots(1)
    rates.plot(ax=axes)
    figs.savefig('rate_data.png')
