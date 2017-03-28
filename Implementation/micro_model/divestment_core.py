
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from itertools import chain
from scipy.integrate import odeint
from scipy.sparse.csgraph import connected_components
from scipy.stats import linregress
from random import shuffle


class Divestment_Core:

    def __init__(self, adjacency=None, opinions=None,
                 investment_clean=None, investment_dirty=None,
                 possible_opinions=None,
                 tau=0.8, phi=.7, eps=0.05,
                 P=100., r_b=0, b_c=1., b_d=1.5, s=0.23, d_c=0.06,
                 b_r0=1., e=10, G_0=3000,
                 R_depletion=True, test=False,
                 C=1, beta=0.06, xi=1. / 8., learning=False,
                 campaign=False):

        # Modes:
        #  1: only economy,
        #  2: economy + opinion formation + decision making,

        if possible_opinions is None:
            possible_opinions = [[0], [1]]

        # check, if heuristic decision making of imitation only
        self.heuristic_decision_making = False
        for p in possible_opinions:
            if p not in [[0], [1]]:
                self.heuristic_decision_making = True

        self.mode = 2

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
        self.cues = {0: self.cue_0, 1: self.cue_1,
                     2: self.cue_2, 3: self.cue_3,
                     4: self.cue_4, 5: self.cue_1}

        # list to save e_trajectory of output variables
        self.e_trajectory = []
        # list to save macroscopic quantities to compare with
        # moment closure / pair based proxy approach
        self.m_trajectory = []
        # list of data for switching events
        self.switchlist = []
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
        self.r_b = r_b
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
        self.possible_opinions = possible_opinions
        # investment_decisions as indices of possible_opinions
        self.opinions = np.array(opinions)
        # to keep track of the current ration of investment_decisions
        self.clean_opinions = np.zeros((len(possible_opinions)))
        self.dirty_opinions = np.zeros((len(possible_opinions)))

        i, n = np.unique(self.opinions,
                         return_counts=True)
        for j in range(len(possible_opinions)):
            if j not in list(i):
                n = np.append(n, [0])
                i = np.append(i, [j])
        self.opinion_state = [n[list(i).index(j)]
                              if j in i else 0
                              for j in range(len(self.possible_opinions))]

        # to keep track of investment decisions.
        self.decision_state = 0.
        # investment decision vector, so far equal to investment_decisions
        if possible_opinions == [[0], [1]]:
            self.investment_decisions = np.array(opinions)
        else:
            self.investment_decisions = np.random.randint(0, 2, self.n)

        # members of ALL household = population   
        self.P = P

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

        # Clean capital depreciation rate
        self.d_c = d_c
        # Dirty capital depreciation rate
        self.d_d = self.d_c
        # knowledge depreciation rate
        self.beta = beta
        # Resource harvest cost per unit (at full resource stock)
        self.b_r0 = b_r0

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
        self.pi = 1. / 2.
        # elasticity of knowledge
        self.xi = xi
        # clean capital elasticity
        self.kappa_c = 1. - self.pi - self.xi
        # dirty capital elasticity
        self.kappa_d = 1. - self.pi
        # fossil->energy->output conversion efficiency (Leontief)
        self.e = e


        # Sector variables

        self.P_c = P / 2.
        self.P_d = P / 2.

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

        self.G = G_0

        if self.e_trajectory_output:
            self.init_e_trajectory()
        if self.m_trajectory_output:
            self.init_m_trajectory()
        if self.switchlist_output:
            self.init_switchlist()

    def cue_0(self, i):
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

    def cue_1(self, i):
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
        if self.r_c>self.r_d*dif:
            dec = 1
        elif self.r_d>self.r_c*dif:
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
        if self.r_c_dot > self.r_d_dot*1.1 and self.r_c > self.d_c:
            dec = 1
        elif self.r_d_dot > self.r_c_dot*1.1 and self.r_d > self.d_c:
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

        neighbors = self.neighbors[:,i].nonzero()[0]
        n_ops = sum(self.investment_decisions[neighbors]) \
                / float(len(neighbors)) if len(neighbors) != 0 else .5
        if n_ops - threshold > 0.5:
            dec = 1
        elif n_ops + threshold < 0.5:
            dec = 0
        else:
            dec = -1

        return dec

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
            # if self.debug:
            #     print self.t, t_max

            # 1 find update candidate and respective update time
            (candidate, neighbor,
             neighbors, update_time) = self.find_update_candidates()

            # 2 integrate economic model until t=update_time:
            self.update_economy(update_time)

            # 3 update opinion formation in case,
            # update candidate was found:
            if candidate >= 0:
                self.update_opinion_formation(candidate,
                                              neighbor, neighbors)

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
                'investment decisions': self.investment_decisions,
                'investment_clean': self.investment_clean,
                'investment_dirty': self.investment_dirty,
                'possible_opinions': self.possible_opinions,
                'tau': self.tau, 'phi': self.phi, 'eps': self.eps,
                'P': self.P, 'r_b': self.r_b, 'b_c': self.b_c,
                'b_d': self.b_d, 's': self.s, 'd_c': self.d_c,
                'b_r0': self.b_r0, 'e': self.e, 'G_0': self.G,
                'C': self.C, 'beta': self.beta, 'xi': self.xi,
                'learning': self.learning,
                'campaign': self.campaign,
                'test': self.debug, 'R_depletion': False}

        if self.converged:
            return 1        # good - consensus reached 
        elif not self.converged and self.R_depletion:
            self.convergence_state = float('nan')
            self.convergence_time = self.t
            return 0        # no consensus found during run time
        elif candidate == -2:
            return -1       # bad run - opinion formation broken
        elif np.isnan(self.G):
            return -2       # bad run - economy broken
        else:
            return -3       # very bad run. Investigations needed

    def b_Rf(self, G):
        """
        Calculates the dependence of resource harvest cost on
        remaining resource stock starts at b_r0 and
        increases with decreasing stock if stock is depleted,
        costs are infinite

        Parameter
        ---------
        G : float
            The quantity of resource remaining in Stock

        Return
        ------
        b_R     : float
            The resource extraction efficiency according to the
            current resource stock
        """

        if G > 0:
            b_R = self.b_r0 * (self.G_0 / G) ** 2
        else:
            b_R = float('inf')
        return b_R

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

        investment_clean = np.where(x0[0:self.n] > 0,
                                    x0[0:self.n],
                                    np.full((self.n), self.epsilon.eps))
        investment_dirty = np.where(x0[self.n:2 * self.n] > 0,
                                    x0[self.n:2 * self.n],
                                    np.full((self.n), self.epsilon.eps))
        P = x0[-3]
        G = x0[-2]
        C = x0[-1]

        K_c = sum(investment_clean)
        K_d = sum(investment_dirty)
        b_R = self.b_Rf(G)

        assert K_c >= 0, 'negative clean capital'
        assert K_d >= 0, 'negative dirty capital'
        assert G >= 0, 'negative resource'
        assert C >= 0, 'negative knowledge'

        X_c = (self.b_c * C ** self.xi * K_c ** self.kappa_c) ** (
        1. / (1. - self.pi))
        X_d = (self.b_d * K_d ** self.kappa_d) ** (1. / (1. - self.pi))
        X_R = (1. - b_R / self.e) ** (1. / (1. - self.pi)) \
            if 1 > b_R / self.e else float('NaN')

        P_c = P * X_c / (X_c + X_d * X_R)
        P_d = P * X_d * X_R / (X_c + X_d * X_R)
        R = 1. / self.e * self.b_d * K_d ** self.kappa_d * P_d ** self.pi

        self.w = self.pi * P ** (self.pi - 1) * (X_c + X_d * X_R) ** (
        1. - self.pi)
        self.r_c = self.kappa_c / \
                   K_c * X_c * P ** self.pi * (X_c + X_d * X_R) ** (- self.pi)
        self.r_d = self.kappa_d / \
                   K_d * X_d * X_R * P ** self.pi * (X_c + X_d * X_R) ** (
        - self.pi)

        # check if dirty sector is profitable (P_d > 0).
        # if not, shut it down.
        if P_d < 0 or np.isnan(X_R):
            P_d = 0
            P_c = P
            R = 0
            self.w = self.b_c * C ** self.xi * K_c ** self.kappa_c * self.pi * P ** (
            self.pi - 1.)
            self.r_c = self.b_c * C ** self.xi * self.kappa_c * \
                       K_c**(self.kappa_c - 1.) * P**self.pi
            self.r_d = 0

        self.R = R
        self.K_c = K_c
        self.K_d = K_d
        self.P = P
        self.P_c = P_c
        self.P_d = P_d
        self.c_R = b_R * R

        self.income = (self.r_c * self.investment_clean
                       + self.r_d * self.investment_dirty
                       + self.w * P / self.n)

        assert all(self.income > 0), \
            'income is negative, X_R: {}, X_d: {}, X_c: {},\
            K_d: {}, K_c: {} \n investment decisions:\
            \n {} \n income: \n {}' \
                .format(X_R, X_d, X_c, K_d, K_c,
                    self.investment_decisions, self.income)

        G_dot = -R if self.R_depletion else 0.0
        P_dot = self.r_b * P
        C_dot = self.b_c * C ** self.xi * P_c ** self.pi \
                * K_c ** self.kappa_c - C * self.beta if self.learning else 0.
        investment_clean_dot = \
            self.investment_decisions \
            * self.s * self.income - self.investment_clean * self.d_c
        investment_dirty_dot = \
            np.logical_not(self.investment_decisions) \
            * self.s * self.income - self.investment_dirty*self.d_d

        x1 = np.fromiter(
            chain.from_iterable([list(investment_clean_dot),
                                 list(investment_dirty_dot),
                                 [P_dot, G_dot, C_dot]]),
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
            list(self.investment_dirty),
            [self.P, self.G, self.C]]), dtype='float')

        # integrate the system unless it crashes.
        if not np.isnan(self.R):
            # with stdout_redirected():
            [x0, x1] = odeint(self.economy_dot_leontief, x0, dt, mxhnil=1)
        else:
            x1 = x0

        self.investment_clean = np.where(x1[0:self.n] > 0,
                                         x1[0:self.n], np.zeros(self.n))
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

        # output economic data
        if self.e_trajectory_output:
            self.update_e_trajectory()
        if self.m_trajectory_output:
            self.update_m_trajectory()

    def find_update_candidates(self):

        i, n = np.unique(self.opinions,
                         return_counts=True)
        self.opinion_state = [n[list(i).index(j)] if j in i
                              else 0
                              for j in range(len(self.possible_opinions))]
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

            # load neighborhood of household i
            neighbors = self.neighbors[:, candidate].nonzero()[0]

            # noise in imitation (exploration of strategies)
            # people trying new stuff at random
            rdn = np.random.uniform()
            if rdn < self.eps*(1-self.phi) and self.imitation:
                old_opinion = self.opinions[candidate]
                new_opinion = np.random.randint(
                    len(self.possible_opinions))
                self.opinions[candidate] = new_opinion
                if old_opinion != new_opinion and self.switchlist_output:
                    self.save_switch(candidate, old_opinion)
                candidate = -1
                break
            # noise in rewiring (sometimes they make new friends
            # at random..)
            elif rdn > 1.-self.eps * self.phi and len(neighbors) > 0:
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
                candidate = -1
                break

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

    def update_opinion_formation(self, candidate, neighbor,
                                 neighbors):

        same_unconnected = np.zeros(self.n, dtype=int)
        opinion = self.opinions

        # adapt or rewire?
        if (self.phi == 1 or (self.phi != 1
                              and np.random.uniform() < self.phi)):
            # if rewire
            for i in xrange(self.n):
                # campaigners rewire to everybody
                if (self.campaign is True and
                        opinion[candidate] == len(self.possible_opinions)):
                    same_unconnected[i] = 1

                # everybody else rewires to people with same opinion
                else:
                    if (opinion[i] == opinion[candidate] and
                            i not in neighbors and i != candidate):
                        same_unconnected[i] = 1
            same_unconnected = same_unconnected.nonzero()[0]
            if len(same_unconnected) > 0:
                new_neighbor = np.random.choice(same_unconnected)
                self.neighbors[candidate, neighbor] = \
                    self.neighbors[neighbor, candidate] = 0
                self.neighbors[candidate, new_neighbor] = \
                    self.neighbors[new_neighbor, candidate] = 1
        else:
            # if adapt
            # compare fitness
            df = self.fitness(neighbor) - self.fitness(candidate)
            # and immitate, if not a campaigner
            if ((self.campaign is False
                    or opinion[candidate] != len(self.possible_opinions)-1)
                    and (np.random.uniform() < .5*(np.tanh(df) + 1))
                    and self.imitation):
                if self.switchlist_output:
                    self.save_switch(candidate, self.opinions[candidate])
                self.opinions[candidate] = self.opinions[neighbor]
        return 0

    def update_decision_making(self):
        """
        Updates the investment decision for all households depending on their
        cue orders (opinion) and the state of the economy
        """

        self.dirty_opinions = np.zeros((len(self.possible_opinions)))
        self.clean_opinions = np.zeros((len(self.possible_opinions)))

        for i in range(self.n):
            for cue in self.possible_opinions[self.opinions[i]]:
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
        self.consensus = all(len(np.unique(opinions[c])) == 1
                             for c in ((cc == i).nonzero()[0]
                             for i in np.unique(cc)))
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

        state = float(sum(opinions))/float(len(opinions))

        attractor = 2./3.
        dist = attractor - state
        # if self.debug == True:
        #     print dist, self.t, self.convergence_state
        alpha = (self.b_r0 / self.e) ** (1. / 2.)

        if self.eps > 0 and dist < 0. and np.isnan(self.convergence_time):
            self.convergence_state = \
                (self.G - alpha * self.G_0) / (self.G_0 * (1. - alpha))
            self.convergence_time = self.t
            self.converged = True

    def fitness(self, agent):
        return self.income[agent]

    def init_e_trajectory(self):
        element = list(chain.from_iterable(
            [['time',
              'wage',
              'r_c',
              'r_d',
              'r_c_dot',
              'r_d_dot',
              'K_c',
              'K_d',
              'P_c',
              'P_d',
              'P',
              'G',
              'R',
              'C',
              'Y_c',
              'Y_d',
              'P_c_cost',
              'P_d_cost',
              'K_c_cost',
              'K_d_cost',
              'c_R',
              'consensus',
              'decision state',
              'G_alpha'],
             [str(x) for x in self.possible_opinions],
             ['c' + str(x) for x in self.possible_opinions],
             ['d' + str(x) for x in self.possible_opinions]]))
        self.e_trajectory.append(element)

        dt = [self.t, self.t]
        x0 = np.fromiter(chain.from_iterable([
            list(self.investment_clean),
            list(self.investment_dirty),
            [self.P, self.G, self.C]]), dtype='float')

        [x0, x1] = odeint(self.economy_dot_leontief, x0, dt)

        self.investment_clean = x1[0:self.n]
        self.investment_dirty = x1[self.n:2 * self.n]
        self.P = x1[-3]
        self.G = x1[-2]
        self.C = x1[-1]

        self.update_e_trajectory()

    def update_e_trajectory(self):
        alpha = (self.b_r0 / self.e) ** (1. / 2.)
        element = list(chain.from_iterable(
            [[self.t,
              self.w,
              self.r_c,
              self.r_d,
              self.r_c_dot,
              self.r_d_dot,
              self.K_c,
              self.K_d,
              self.P_c,
              self.P_d,
              self.P,
              self.G,
              self.R,
              self.C,
              self.Y_c,
              self.Y_d,
              self.P_c * self.w,
              self.P_d * self.w,
              self.K_c * self.r_c,
              self.K_d * self.r_d,
              self.c_R,
              self.converged,
              self.decision_state,
              (self.G - alpha * self.G_0) / (self.G_0 * (1. - alpha))],
             self.opinion_state,
             self.clean_opinions,
             self.dirty_opinions]))
        self.e_trajectory.append(element)

    def get_e_trajectory(self):
        # make up DataFrame from micro data
        columns = self.e_trajectory.pop(0)
        df = pd.DataFrame(self.e_trajectory, columns=columns)
        df = df.set_index('time')

        return df

    def init_m_trajectory(self):
        """
        This function initializes the e_trajectory for the output of the
        macroscopic quantitites as computed via moment closure and
        pair based proxy.
        :return: None
        """
        element = ['time', 'x', 'y', 'z', 'mucc', 'mucd', 'mudc', 'mudd', 'c',
                   'g']
        self.m_trajectory.append(element)

        dt = [self.t, self.t]
        x0 = np.fromiter(chain.from_iterable([
            list(self.investment_clean),
            list(self.investment_dirty),
            [self.P, self.G, self.C]]), dtype='float')

        [x0, x1] = odeint(self.economy_dot_leontief, x0, dt)

        self.investment_clean = x1[0:self.n]
        self.investment_dirty = x1[self.n:2 * self.n]
        self.P = x1[-3]
        self.G = x1[-2]
        self.C = x1[-1]

        self.update_m_trajectory()

    def update_m_trajectory(self):
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
        d = - self.investment_decisions + 1

        n = self.n
        k = float(sum(sum(self.neighbors))) / 2

        nc = sum(self.investment_decisions)
        nd = sum(- self.investment_decisions + 1)

        cc = cl(adj, c, c) / 2
        cd = cl(adj, c, d)
        dd = cl(adj, d, d) / 2

        x = float(nc - nd) / n
        y = float(cc - dd) / k
        z = float(cd) / k

        mucc = sum(self.investment_decisions * self.investment_clean) / nc
        mucd = sum(self.investment_decisions * self.investment_dirty) / nc
        mudc = sum((1 - self.investment_decisions)
                   * self.investment_clean) / nd
        mudd = sum((1 - self.investment_decisions)
                   * self.investment_dirty) / nd

        entry = [self.t, x, y, z, mucc, mucd, mudc, mudd, self.C / n,
                 self.G / n]
        self.m_trajectory.append(entry)

    def get_m_trajectory(self):
        # make up Dataframe from macro data:
        columns = self.m_trajectory.pop(0)
        df = pd.DataFrame(self.m_trajectory, columns=columns)
        df = df.set_index('time')

        return df

    def init_switchlist(self):
        """Initializes the switchlist by naming the collumns"""
        self.switchlist = [['time', '$K^{(c)}$', '$K^{(d)}$', 'direction']]

    def save_switch(self, i, direction):
        """
        Adds an entry to the switchlist.

        Parameters
        ----------
        i : int
            the index of the household that switched its opinion
        direction : int
            the direction that it switched in
        """
        self.switchlist.append([self.t,
                                self.investment_clean[i],
                                self.investment_dirty[i],
                                direction])

    def get_switch_list(self):
        columns = self.switchlist.pop(0)
        df = pd.DataFrame(self.switchlist, columns=columns)
        df = df.set_index('time')

        return df


if __name__ == '__main__':
    """
    Perform test run and plot some output to check
    functionality
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as mp

    output_location = 'test_output/'\
        + datetime.datetime.now().strftime("%d_%m_%H-%M-%Ss") + '_output'

    # Initial conditions:
    FFH = False

    if FFH:

        nopinions = [10, 10, 10, 10, 10, 10, 10, 10]
        possible_opinions = [[2, 3],  # short term investor
                             [3, 2],  # long term investor
                             [4, 2],  # short term herder
                             [4, 3],  # trending herder
                             [4, 1],  # green conformer
                             [4, 0],  # dirty conformer
                             [1],  # gutmensch
                             [0]]  # redneck
        input_parameters = {'tau': 1, 'eps': 0.05, 'b_d': 1.2,
                            'b_c': 1., 'phi': 0.8, 'e': 100,
                            'G_0': 1500, 'b_r0': 0.1 ** 2 * 100,
                            'possible_opinions': possible_opinions,
                            'C': 1, 'xi': 1. / 8., 'beta': 0.06,
                            'campaign': False, 'learning': True}

    if not FFH:
        # investment_decisions:
        nopinions = [10, 10]
        possible_opinions = [[0], [1]]

        # Parameters:

        input_parameters = {'tau': 1, 'eps': 0.05, 'b_d': 1.2,
                            'b_c': 1., 'phi': 0.8, 'e': 100,
                            'G_0': 1500, 'b_r0': 0.1 ** 2 * 100,
                            'possible_opinions': possible_opinions,
                            'C': 1, 'xi': 1. / 8., 'beta': 0.06,
                            'campaign': False, 'learning': True}

    cops = ['c'+str(x) for x in possible_opinions]
    dops = ['d'+str(x) for x in possible_opinions]
    colors = [np.random.rand(3, 1) for x in possible_opinions]
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

    init_conditions = (adjacency_matrix, opinions,
                       clean_investment, dirty_investment)


    # Initialize Model

    model = Divestment_Core(*init_conditions,
                            **input_parameters)

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

    print connected_components(model.neighbors, directed=False)
    print 'investment decisions:'
    print model.investment_decisions
    print 'consensus reached?', model.converged
    print model.convergence_state
    print 'finish time', model.t
    print 'steps computed', model.steps

    colors = [c for c in 'gk']

    df = model.get_e_trajectory()
    print df.columns

    fig = mp.figure()
    ax1 = fig.add_subplot(221)
    df[['r_c', 'r_d']].plot(ax=ax1, style=colors)

    ax2 = fig.add_subplot(223)
    df[['wage']].plot(ax=ax2)

    ax3 = fig.add_subplot(224)
    df[['K_c', 'K_d']].plot(ax=ax3, style=colors)

    ax4 = fig.add_subplot(222)
    df[['G']].plot(ax=ax4, style=colors[1])
    ax5 = ax4.twinx()
    df[['C']].plot(ax=ax5, style=colors[0])

    fig.tight_layout()
    mp.savefig('example_trajectory.png')
