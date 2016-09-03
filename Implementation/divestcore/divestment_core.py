
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from itertools import chain
from scipy.integrate import odeint
from scipy.sparse.csgraph import connected_components
from scipy.stats import linregress
from random import shuffle


class divestment_core:

    def __init__(self, adjacency=None, opinions=None,
                 investment_clean=None, investment_dirty=None,
                 possible_opinions=[[0], [1]],
                 tau=0.8, phi=.7, eps=0.05,
                 P=1000., r_b=0, b_c=1., b_d=1.5, s=0.23, d_c=0.06,
                 b_R0=1., e=50, G_0=1000,
                 R_depletion=True, test=False):

        # Modes:
        #  1: only economy,
        #  2: economy + opinion formation + decision making,

        self.mode = 2

        # General Parameters

        self.debug = test               # turn output for debugging on or off
        self.trajectory_output = True   # toggle trajectory output
        self.run_full_time = True       # toggle whether to run full time or only until consensus 
        self.R_depletion = R_depletion  # toggle resource depletion
        self.imitation = True           # toggle imitation in avm
        self.epsilon = np.finfo(dtype='float')

        # General Variables

        self.t = 0                              # System Time
        self.steps = 0                          # Step counter for output
        self.converged = False                  # eps == 0: 0 for no consensus, 1 consensus
                                                # eps>0 0 for no convergence, 1 for convergence at t_max
        self.convergence_time = float('NaN')    # safes the system time at which consensus is reached
        self.convergence_state = -1             # eps==0: -1 for no consensus, 1 for clean consensus, 
                                                # 0 for dirty consensus, in between for fragmentation
                                                # eps>0: if converged: opinion state at time of convergence
                                                # if not converged: opinion state at t_max

        self.cues = {0: self.cue_0, 1:self.cue_1, 
                2:self.cue_2, 3:self.cue_3, 
                4:self.cue_4}                   # dictionary of decision cues

        self.trajectory = []                    # list to save trajectory of output variables
        self.final_state = {}                   # dictionary for final state

        # Household parameters

        self.tau = tau                  # mean waiting time between social updates
        self.phi = phi                  # rewiring probability for adaptive voter model
        self.eps = eps                  # percentage of rewiring and imitation events that are noise

        self.N = adjacency.shape[0]     # number of households
        self.r_b = r_b                  # birth rate for household members
        self.s = s                      # percentage of income saved

        # Decision making variables:

        self.N_mem = 10                       # number of steps that households memorize to estimate trend
        self.r_cs = []                       # memory of r_c values
        self.r_ds = []                       # memory of r_d values
        self.t_rs = []                       # times of memories

        # Household variables

        # Individual

        self.waiting_times = \
                np.random.exponential(scale=self.tau, size=self.N)  # waiting times between rewiring events for each household
        self.neighbors = adjacency                  # adjacency matrix between households
        self.possible_opinions = possible_opinions  # to select random opinions, 
                                                    # all possible opinions must be known
        self.opinions = np.array(opinions)          # opinions as indices of possible_opinions

                                                    # to keep track of the current ration of opinions
        ### these two need some scrutiny
        i, n = np.unique(self.opinions, \
                return_counts = True)
        self.opinion_state = [n[list(i).index(j)] if j in i else 0 for j in range(len(self.possible_opinions))] 

        self.decision_state = 0.                    # to keep track of investment decisions.
        self.investment_decisions = \
                np.random.randint(0,2,self.N)       # investment decision vector, so far equal to opinions

        # members of ALL household = population   
        self.P = P

        # household investment in dirty capital
        if investment_dirty is None:
            self.investment_dirty = np.ones((self.N))
        else:
            self.investment_dirty = investment_dirty

        # household investment in clean capital
        if investment_clean is None:
            self.investment_clean = np.ones((self.N))
        else:
            self.investment_clean = investment_clean

        # household income (for social update)
        self.income = np.zeros((self.N))

        # Aggregated
        self.K_c = 0    # total clean capital (supply)
        self.K_d = 0    # total dirty capital (supply)
        self.K = self.K_c + self.K_d

        # Sector parameters

        self.d_c = d_c              # Clean capital depreciation rate
        self.d_d = self.d_c         # Dirty capital depreciation rate
        self.b_R0 = b_R0            # Resource harvest cost per unit (at full resource stock)

        # for Cobb Douglas economy
        # elasticities of labor and resource use are fixed
        # (pi = 2/5, rho = 3/4, epsilon = 5/4)
        # to be able to solve market clearing analytically

        # for Leontief dirty sector without profits,
        # capital and labor elasticities must be equal
        # in both sectors and satisfy pi + kappa = 1

        self.b_c = b_c                   # solow residual for clean sector
        self.b_d = b_d                   # solow residual for dirty sector

        self.pi = 1./2.                 # labor elasticity (equal in both sectors)
        self.kappa_c = 1. - self.pi     # clean capital elasticity
        self.kappa_d = 1. - self.pi     # dirty capital elasticity
        self.rho = 3./4.                # fossil resource elasticity (Cobb-Douglas)
        self.e = 10.                    # fossil->energy->output conversion efficiency (Leontief)

        # Sector variables

        self.P_c = P/2.
        self.P_d = P/2.

        self.K_c = 0.
        self.K_d = 0.

        self.R = 1.

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

        self.X_R = 1.

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
        if self.r_c>self.r_d*1.1:
            dec = 1
        elif self.r_d>self.r_c*1.1:
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
        if self.r_c_dot>self.r_d_dot*1.1:
            dec = 1
        elif self.r_d_dot>self.r_c_dot*1.1:
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
        n_ops = sum(self.investment_decisions[neighbors])/float(len(neighbors)) if len(neighbors)!=0 else .5
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

        if self.trajectory_output:
            self.init_economic_trajectory()

        if self.mode == 1:
            # only economic model:

            while self.t < t_max:
                # 1 increase update_time:
                update_time = self.t + 0.001*t_max

                # 2 integrate economic model until t=update_time
                self.update_economy(update_time)

                # 3 update investment decision making:
                self.update_decision_making()

                # 4 check for 2/3 majority for clean investment
                self.detect_convergence(self.investment_decisions)

        elif self.mode == 2:
            # economic, opinion formation and decision making model:

            while self.t < t_max:
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
                'investment_clean': self.investment_clean,
                'investment_dirty': self.investment_dirty,
                'possible_opinions': self.possible_opinions,
                'tau': self.tau, 'phi': self.phi, 'eps': self.eps,
                'P': self.P, 'r_b': self.r_b, 'b_c': self.b_c,
                'b_d': self.b_d, 's': self.s, 'd_c': self.d_c,
                'b_R0': self.b_R0, 'e': self.e, 'G_0': self.G,
                'test': self.debug, 'R_depletion': False}

        if self.converged:
            return 1        # good - consensus reached
        elif not self.converged:
            self.convergence_state = self.opinion_state
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
        remaining resource stock starts at b_R0 and 
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
        
        if G>0:
            b_R = self.b_R0*(self.G_0/G)**2
        else:
            b_R = float('inf')
        return b_R

    def economy_dot(self, x0, t):

        investment_clean = x0[0:self.N]
        investment_dirty = x0[self.N:2*self.N]
        P = x0[-2]
        G = x0[-1]

        K_c = sum(investment_clean)
        K_d = sum(investment_dirty)
        b_R = self.b_Rf(G)

        X_c = (self.b_c * K_c**self.kappa_c)**(-5./3.)
        X_d = (self.b_d * K_d**self.kappa_d)**(-5./3.)
        X_R = ((3. * self.b_d * K_d**self.kappa_d)/(5. * b_R))**(2.)

        P_c = (X_d/X_c) * X_R**(-5./4.)
        P_d = P - (X_d/X_c) * X_R**(-5./4.)
        R   = X_R*(P - (X_d/X_c)*X_R**(-5./4.))**(4./5.)

        #Here comes a dirty hack to solve a numerical problem.
        #Find a better solution if possible some time.
        #Well, actually, I am not so sure, if this is a bug.
        #It might just be the demand for clean labor exceeding the
        #total labor supply. In this case, this hack would be the 
        #correct solution to the market clearing equations.
        if P_d < 0:
            P_d = 0
            P_c = P
            R = 0

        self.w   = (2./5.) * X_d**(-3./5.) * X_R**(-3./4.)
        self.r_c = self.kappa_c / (X_c * K_c) * X_d**(2./5.) * X_R**(-2.)
        self.r_d = self.kappa_d / K_d * X_d**(-3./5.) * X_R**(3./4.) * (P_d)

        self.R = R
        self.K_c = K_c
        self.K_d = K_d
        self.P   = P
        self.P_c = P_c
        self.P_d = P_d
        self.c_R = b_R * R

        self.income = self.r_c*self.investment_clean \
                + self.r_d*self.investment_dirty \
                + self.w*P/self.N

        G_dot = -R
        P_dot= self.r_b * P
        investment_clean_dot = self.investment_decisions*self.s*self.income \
                - self.investment_clean*self.d_c
        investment_dirty_dot = np.logical_not(self.investment_decisions)*self.s*self.income \
                - self.investment_dirty*self.d_d

        x1 = np.fromiter(chain.from_iterable([ 
            list(investment_clean_dot), 
            list(investment_dirty_dot), 
            [P_dot, G_dot]]), dtype='float')

        return x1

    def economy_dot_leontief(self , x0, t):

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
            clean household investments [0:N] and
            dirty household investments [N:2N].
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
            of household members (depreciated) [0:N]
            clean household investments [N:2N] and
            dirty household investments [2N:3N].
            The last entry is the remaining fossil
            reserves.
        """

        investment_clean = np.where(x0[0:self.N] > 0,
                                    x0[0:self.N],
                                    np.full((self.N), self.epsilon.eps))
        investment_dirty = np.where(x0[self.N:2*self.N] > 0,
                                    x0[self.N:2*self.N],
                                    np.full((self.N), self.epsilon.eps))
        P = x0[-2]
        G = x0[-1]

        K_c = sum(investment_clean)
        K_d = sum(investment_dirty)
        b_R = self.b_Rf(G)

        assert K_c >= 0, 'negative clean capital'
        assert K_d >= 0, 'negative dirty capital'
        assert G >= 0, 'negative resource'

        X_c = (self.b_c * K_c**self.kappa_c)**(1./(1.-self.pi))
        X_d = (self.b_d * K_d**self.kappa_d)**(1./(1.-self.pi))
        X_R = (1. - b_R/self.e)**(1./(1.-self.pi))\
            if 1 > b_R/self.e else float('NaN')

        P_c = P*X_c/(X_c + X_d*X_R)
        P_d = P*X_d*X_R/(X_c + X_d*X_R)
        R = 1./self.e * self.b_d * K_d**self.kappa_c * P_d**self.pi

        self.w = self.pi * P**(self.pi - 1) * (X_c + X_d*X_R)**(1.-self.pi)
        self.r_c = self.kappa_c /\
            K_c * X_c * P**self.pi * (X_c + X_d*X_R)**(-self.pi)
        self.r_d = self.kappa_d /\
            K_d * X_d * X_R * P**self.pi * (X_c + X_d*X_R)**(-self.pi)

        # check if dirty sector is profitable (P_d > 0).
        # if not, shut it down.
        if P_d < 0 or np.isnan(X_R):
            P_d = 0
            P_c = P
            R = 0
            self.w = self.b_c * K_c**self.kappa_c * self.pi * P**(self.pi - 1.)
            self.r_c = self.b_c * self.kappa_c *\
                K_c**(self.kappa_c - 1.) * P**self.pi
            self.r_d = 0

        self.R = R
        self.K_c = K_c
        self.K_d = K_d
        self.P = P
        self.P_c = P_c
        self.P_d = P_d
        self.c_R = b_R * R
        self.X_R = X_R

        self.income = (self.r_c*self.investment_clean
                       + self.r_d*self.investment_dirty
                       + self.w*P/self.N)
        assert all(self.income > 0),\
                'income is negative, X_R: {}, X_d: {}, X_c: {}, K_d: {}, K_c: {}\
                \n investment decisions: \n {} \n income: \n {}'\
            .format(X_R, X_d, X_c, K_d, K_c,
                    self.investment_decisions, self.income)

        G_dot = -R if self.R_depletion else 0.0
        P_dot = self.r_b * P
        investment_clean_dot = self.investment_decisions*self.s*self.income \
            - self.investment_clean*self.d_c
        investment_dirty_dot = np.logical_not(self.investment_decisions)\
            * self.s * self.income - self.investment_dirty*self.d_d

        x1 = np.fromiter(chain.from_iterable([
                                             list(investment_clean_dot),
                                             list(investment_dirty_dot),
                                             [P_dot, G_dot]]), dtype='float')

        return x1

    def update_economy(self, update_time):

        dt = [self.t, update_time]
        x0 = np.fromiter(chain.from_iterable([
            list(self.investment_clean),
            list(self.investment_dirty),
            [self.P, self.G]]), dtype='float')

        # integrate the system unless it crashes.
        if not np.isnan(self.R):
            # with stdout_redirected():
            [x0, x1] = odeint(self.economy_dot_leontief, x0, dt, mxhnil=1)
        else:
            x1 = x0

        self.investment_clean = np.where(x1[0:self.N] > 0,
                                         x1[0:self.N], np.zeros(self.N))
        self.investment_dirty = np.where(x1[self.N:2*self.N] > 0,
                                         x1[self.N:2*self.N], np.zeros(self.N))
        self.P = x1[-2]
        self.G = x1[-1]

        # memorize return rates for trend estimation
        self.r_cs.append(self.r_c)
        self.r_ds.append(self.r_d)
        self.t_rs.append(self.t)
        if len(self.r_cs) > self.N_mem:
            self.r_cs.pop(0)
            self.r_ds.pop(0)
            self.t_rs.pop(0)

        self.r_c_dot = linregress(self.t_rs, self.r_cs)[0]
        self.r_d_dot = linregress(self.t_rs, self.r_ds)[0]
        if np.isnan(self.r_c_dot):
            print self.r_cs
            self.r_c_dot = 0
        if np.isnan(self.r_d_dot):
            print self.r_ds
            self.r_d_dot = 0
        self.t = update_time
        self.steps += 1

        # calculate market shares:
        self.Y_c = self.b_c*self.K_c**self.kappa_c*self.P_c**self.pi
        self.Y_d = self.b_c*self.K_d**self.kappa_d*self.P_d**self.pi*self.R**self.rho

        # output economic data
        if self.trajectory_output:
            self.update_economic_trajectory()

    def find_update_candidates(self):

        # For prototyping, use reduced opinion formation with only
        # investment decision outcomes as opinion.

        opinions = self.opinions
        i, n = np.unique(self.opinions,
                         return_counts=True)
        self.opinion_state = [n[list(i).index(j)] if j in i
                              else 0
                              for j in range(len(self.possible_opinions))]
        i = 0
        i_max = 1000*self.N
        neighbor = self.N
        while i < i_max:

            # find household with min waiting time
            candidate = self.waiting_times.argmin()

            # remember update_time and increase waiting time of
            # household
            update_time = self.waiting_times[candidate]
            self.waiting_times[candidate] += \
                np.random.exponential(scale=self.tau)

            # load neighborhood of household i
            neighbors = self.neighbors[:, candidate].nonzero()[0]

            # noise in imitation (exploration of strategies)
            rdn = np.random.uniform()
            if rdn < self.eps*(1-self.phi) and self.imitation:
                self.opinions[candidate] = np.random.randint(
                        len(self.possible_opinions))
                candidate = -1
                break
            # noise in rewiring (sometimes they make new friends
            # at random..)
            elif rdn > 1.-self.eps * self.phi and len(neighbors) > 0:
                unconnected = np.zeros(self.N, dtype=int)
                for i in range(self.N):
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
                if (opinions[candidate] == opinions[neighbor]):

                    # if candidate and neighbor have same
                    # preferences, they
                    # not suitable for update. (RETRY)
                    neighbor = self.N

            if neighbor < self.N:
                # update candidate found (GOD)
                break
            elif self.converged:
                candidate = -1
                update_time = self.t + self.tau
                break
            else:
                i += 1
                if i % self.N == 0:
                    if self.detect_consensus_state(opinions):
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

        same_unconnected = np.zeros(self.N, dtype=int)
        opinion = self.investment_decisions

        # adapt or rewire?
        if (self.phi == 1 or (self.phi != 1
                              and np.random.uniform() < self.phi)):
            # if rewire
            for i in xrange(self.N):
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
            if (np.random.uniform() < .5*(np.tanh(df) + 1)) and self.imitation:
                    self.opinions[candidate] = \
                            self.opinions[neighbor]
        return 0

    def update_decision_making(self):
        # update decision vector for all
        # households depending on their
        # preferences and the state of the
        # economy

        for i in range(self.N):
            for cue in self.possible_opinions[self.opinions[i]]:
                decision = self.cues[cue](i)
                if decision != -1:
                    self.investment_decisions[i] = decision
                    break
            if decision == -1:
                self.investment_decisions[i] = np.random.randint(2)

        self.decision_state = np.mean(self.investment_decisions)

        return 0

    def detect_consensus_state(self, opinions):
        # check if network is split in components with
        # same opinions/preferences
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
        to eps/2(1-phi) which is the equilibrium state
        if all imitation events are d->c.
        If the system converged, set convergence time.

        Parameters:
        -----------
        opinions: [int]
            list of opinions.
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
        alpha = (self.b_R0/self.e)**(1./2.)

        if self.eps > 0 and dist < 0. and np.isnan(self.convergence_time):
            self.convergence_state =\
                (self.G-alpha*self.G_0)/(self.G_0*(1.-alpha))
            self.convergence_time = self.t
            self.convergence = True

    def fitness(self, agent):
        return self.income[agent]

    def update_economic_trajectory(self):
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
              self.Y_c,
              self.Y_d,
              self.P_c*self.w,
              self.P_d*self.w,
              self.K_c*self.r_c,
              self.K_d*self.r_d,
              self.c_R,
              self.converged,
              self.decision_state],
             self.opinion_state]))
        self.trajectory.append(element)

    def init_economic_trajectory(self):
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
              'Y_c',
              'Y_d',
              'P_c_cost',
              'P_d_cost',
              'K_c_cost',
              'K_d_cost',
              'c_R',
              'consensus',
              'decision state'],
             [str(x) for x in self.possible_opinions]]))
        self.trajectory.append(element)

        dt = [self.t, self.t]
        x0 = np.fromiter(chain.from_iterable([
            list(self.investment_clean),
            list(self.investment_dirty),
            [self.P, self.G]]), dtype='float')

        [x0, x1] = odeint(self.economy_dot_leontief, x0, dt)

        self.investment_clean = x1[0:self.N]
        self.investment_dirty = x1[self.N:2*self.N]
        self.P = x1[-2]
        self.G = x1[-1]

        self.update_economic_trajectory()




if __name__ == '__main__':
    """
    Perform test run and plot some output to check
    functionality
    """
    import pandas as pd
    import matplotlib.pyplot as mp
   
   
    output_location = 'test_output/' + datetime.datetime.now().strftime("%d_%m_%H-%M-%Ss") + '_output'


    # Initial conditions:

    # opinions:

    nopinions = [20,20,0,0,0]
    possible_opinions = [[1,3],[2,3],[2,4],[3,4],[4]]

    opinions = []
    for i, n in enumerate(nopinions):
        opinions.append(np.full((n),i, dtype='I'))
        print i, n
        print opinions
    opinions = [item for sublist in opinions for item in sublist]
    shuffle(opinions)

    # network:

    N = sum(nopinions)
    p = .2

    while True:
        net = nx.erdos_renyi_graph(N, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()
    print adjacency_matrix
    print connected_components(adjacency_matrix)
    
    init_conditions = (adjacency_matrix, opinions)

    # Parameters:

    input_parameters = {'tau':0.1, 'eps':0.05, 'b_d':1.6,\
            'G_0':3000, 'possible_opinions':possible_opinions}

    # Initialize Model

    model = divestment_core(*init_conditions, tau = 0.1, eps=0.05, b_d = 1.6, G_0 = 3000)

    # Turn on debugging

    model.debug = True

    # Run Model

    model.run(t_max=50)

    # Print some output

    print connected_components(model.neighbors, directed=False)
    print 'investment decisions:'
    print model.investment_decisions
    print 'consensus reached?', model.converged
    print model.convergence_state
    print 'finish time', model.t
    print 'steps computed', model.steps



#   trj = model.trajectory
#   headers = trj.pop(0)
#   df = pd.DataFrame(trj, columns=headers)
#   df = df.set_index('time')
#
#   fig = mp.figure()
#   ax1 = fig.add_subplot(231)
#   df[['K_d', 'K_c']].plot(ax = ax1)
#   ax1.set_yscale('log')
#  
#   ax2 = fig.add_subplot(232)
#   df[['P_d', 'P_c']].plot(ax = ax2)
#   mp.axvline(model.convergence_time)
#   ax2.set_yscale('log')
#
#   ax3 = fig.add_subplot(233)
#   df[['G']].plot(ax = ax3)
#   mp.axvline(model.convergence_time)
#   ax3.set_yscale('log')
#
#   ax4 = fig.add_subplot(234)
#   df[['r_c_dot', 'r_d_dot']].plot(ax = ax4)
#   mp.axvline(model.convergence_time)
#
#   ax5 = fig.add_subplot(235)
#   df[['r_d', 'r_c']].plot(ax = ax5)
#   mp.axvline(model.convergence_time)
#   #ax5.set_yscale('log')
#
#   ax6 = fig.add_subplot(236)
#   df[['decision state']].plot(ax = ax6)
#   mp.axvline(model.convergence_time)
#   ax6.set_yscale('log')
#
#   mp.show()
#   print df[['P_c', 'P_d', 'G', 'K_c', 'K_d', 'r_c_dot', 'r_d_dot']]


