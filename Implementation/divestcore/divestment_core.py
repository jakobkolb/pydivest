

import os
import datetime 
import csv
import numpy as np
import pandas as pd
import networkx as nx
from itertools import chain
from scipy.integrate import odeint
from scipy.sparse.csgraph import connected_components
from stdout_redirected import stdout_redirected
from math import exp



class divestment_core:

    def __init__(self, adjacency, opinions, P_start=1000., tau=0.8, phi=.7, eps=0.05, possible_opinions = [0,1]):

        ## Modes: 1: only economy, 2: + opinion formation, 3: + decision heuristics

        self.mode = 2
        
        ## General Parameters

        self.debug = False              #turn output for debugging on or off
        self.trajectory_output = True   #toggle trajectory output
        self.run_full_time = True       #toggle whether to run full time or only until consensus 
        
        ## General Variables

        self.t = 0                              # System Time
        self.steps = 0                          # Step counter for output
        self.consensus = False                  # 0 for no consensus, 1 consensus
        self.consensus_state = -1               # -1 for no consensus, 1 for clean consensus, 
                                                # 0 for dirty consensus, in between for fragmentation
        self.consensus_time = float('NaN')      # safes the system time at which consensus is reached
        self.opinions = opinions
        self.opinion_state = np.mean(opinions) # to keep track of the current ration of opinions
        self.possible_opinions = possible_opinions
        self.trajectory = []                    # list to save trajectory of output variables

        ## Household parameters

        self.tau = tau                  #mean waiting time between social updates
        self.phi = phi                  #rewiring probability for adaptive voter model
        self.eps = eps

        self.N = adjacency.shape[0]                 #number of households
        self.net_birth_rate = .0                   #birth rate for household members
        self.s = .23                     #percentage of income consumed

        ## Household variables

        ## Individual
        self.P = P_start                            #members of ALL household = population
        self.investment_dirty = np.ones((self.N))   #household investment in clean capital
        self.investment_clean = np.ones((self.N))   #household investment in dirty capital
        self.waiting_times = np.zeros((self.N))      
        self.waiting_times = \
                np.random.exponential(scale=self.tau, size=self.N)  #waiting times between rewiring events for each household
        self.neighbors = adjacency                  #adjacency matrix between households
        self.investment_decision = opinions         #investment decision vector
        self.income = np.zeros((self.N))            #household income (for social update)

        ## Aggregated
        self.K_c = 0    #total clean capital (supply)
        self.K_d = 0    #total dirty capital (supply)
        self.K = self.K_c + self.K_d

        ## Sector parameters

        self.d_c = .06              # Clean capital depreciation rate
        self.d_d = self.d_c     # Dirty capital depreciation rate
        self.b_R = 1.                   # Resource harvest cost per unit (at full resource stock)
        self.b_r_present = self.b_R     # Kept for upward compatibility

        ## for Cobb Douglas economy 
        ## elasticities of labor and resource use are fixed 
        ## (pi = 2/5, rho = 3/4, epsilon = 5/4) 
        ## to be able to solve market clearing analytically

        ## for Leontief dirty sector without profits,
        ## capital and labor elasticities must be equal
        ## in both sectors and satisfy pi + kappa = 1

        self.b_c = 1.                   # solow residual for clean sector
        self.b_d = 3.                   # solow residual for dirty sector

        self.pi = 2./5.                 # labor elasticity (equal in both sectors)
        self.kappa_c = 1. - self.pi     # clean capital elasticity
        self.kappa_d = 1. - self.pi     # dirty capital elasticity
        self.rho = 3./4.                # fossil resource elasticity (Cobb-Douglas)
        self.e = 10.                    # fossil->energy->output conversion efficiency (Leontief)

        ## Sector variables

        self.P_c = P_start/2.
        self.P_d = P_start/2.

        self.K_c = 0.
        self.K_d = 0.

        self.R = 1.

        self.Y_c = 0.
        self.Y_d = 0.

        ## derived Sector variables

        self.w = 0.
        self.r_c = 0.
        self.r_d = 0.
        self.c_R = 0

        ## Ecosystem parameters

        self.G_0 = 10000. 

        ## Ecosystem variables

        self.G = self.G_0

        self.X_R = 1.

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
            if exit_status == 1: consensus reached
            if exit_status == 0: no consensus reached at t=t_max
            if exit_status ==-1: no consensus & no update candidates found (BAD)
            if exit_status ==-2: economic model broken (BAD)
        """

        if self.trajectory_output: 
            self.init_economic_trajectory()

        if self.mode == 1:
            while self.t<t_max:
                # Update economy time is up
                candidate, neighbor, neighbors, update_time = self.find_update_candidates()
                if self.run_full_time and self.consensus:
                    update_time += t_max/100.
                self.update_economy(update_time)

                if not self.run_full_time and self.consensus:
                    break

        elif self.mode == 2:
            while self.t<t_max:
                # Update economy, social and decision making
                candidate, neighbor, neighbors, update_time = self.find_update_candidates()

                self.update_economy(update_time)
                if candidate >= 0: self.update_opinion_formation(candidate, neighbor, neighbors)
                self.update_decision_making()

                if not self.run_full_time and self.consensus:
                    break

        if candidate==-1: 
            return 1        #good - consensus reached
        elif candidate==-2: 
            return -1       #bad run - opinion formation broken
        elif np.isnan(self.G): 
            return -2       #bad run - economy broken
        else: 
            return 0        # no consensus found during run time

    def b_r(self, G):
        """
        Calculates the dependence of resource harvest cost on 
        remaining resource stock starts at b_r_present and 
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
            b_R = self.b_r_present*(self.G_0/G)**2
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
        b_R = self.b_r(G)

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
        P_dot= self.net_birth_rate * P
        investment_clean_dot = self.investment_decision*self.s*self.income \
                - self.investment_clean*self.d_c
        investment_dirty_dot = np.logical_not(self.investment_decision)*self.s*self.income \
                - self.investment_dirty*self.d_d

        x1 = np.fromiter(chain.from_iterable([ 
            list(investment_clean_dot), 
            list(investment_dirty_dot), 
            [P_dot, G_dot]]), dtype='float')

        return x1

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

        investment_clean = x0[0:self.N]
        investment_dirty = x0[self.N:2*self.N]
        P = x0[-2]
        G = x0[-1]

        K_c = sum(investment_clean)
        K_d = sum(investment_dirty)
        b_R = self.b_r(G)

        X_c = (self.b_c * K_c**self.kappa_c)**(1./(1.-self.pi))
        X_d = (self.b_d * K_d**self.kappa_d)**(1./(1.-self.pi))
        X_R = (1. - b_R/self.e)**(1./(1.-self.pi)) if 1 > b_R/self.e else float('NaN')

        P_c = P*X_c/(X_c + X_d*X_R)
        P_d = P*X_d*X_R/(X_c + X_d*X_R)
        R   = 1./self.e * self.b_d * K_d**self.kappa_c * P_d**self.pi

        self.w   = self.pi * P**(self.pi - 1) * (X_c + X_d*X_R)**(1.-self.pi)
        self.r_c = self.kappa_c / K_c * X_c * P**self.pi * (X_c + X_d*X_R)**(-self.pi)
        self.r_d = self.kappa_d / K_d * X_d * X_R * P**self.pi * (X_c + X_d*X_R)**(-self.pi)

        # check if dirty sector is profitable (P_d > 0).
        # if not, shut it down.
        if P_d < 0 or np.isnan(X_R):
            P_d = 0
            P_c = P
            R = 0
            self.w = self.b_c * K_c**self.kappa_c * self.pi * P**(self.pi - 1.)
            self.r_c = self.b_c * self.kappa_c * K_c**(self.kappa_c - 1.) * P**self.pi
            self.r_d = 0

        self.R = R
        self.K_c = K_c
        self.K_d = K_d
        self.P   = P
        self.P_c = P_c
        self.P_d = P_d
        self.c_R = b_R * R
        self.X_R = X_R

        self.income = self.r_c*self.investment_clean \
                + self.r_d*self.investment_dirty \
                + self.w*P/self.N

        G_dot = -R
        P_dot= self.net_birth_rate * P
        investment_clean_dot = self.investment_decision*self.s*self.income \
                - self.investment_clean*self.d_c
        investment_dirty_dot = np.logical_not(self.investment_decision)*self.s*self.income \
                - self.investment_dirty*self.d_d

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

        #integrate the system unless it crashes.
        if not np.isnan(self.R):
            #with stdout_redirected():
            [x0,x1] = odeint(self.economy_dot_leontief, x0, dt, mxhnil=1)
        else:
            x1 = x0

        self.investment_clean = x1[0:self.N]
        self.investment_dirty = x1[self.N:2*self.N]
        self.P = x1[-2]
        self.G = x1[-1]

        self.t = update_time
        self.steps += 1

        #calculate market shares:
        self.Y_c = self.b_c*self.K_c**self.kappa_c*self.P_c**self.pi
        self.Y_d = self.b_c*self.K_d**self.kappa_d*self.P_d**self.pi*self.R**self.rho

        #output economic data
        if self.trajectory_output:
            self.update_economic_trajectory()

    def find_update_candidates(self):

        #For prototyping, use reduced opinion formation with only
        #investment decision outcomes as opinion.

        #THIS MUST BE UPDATED IF CUE ORDERS ARE USED AS OPINIONS
        opinions = self.investment_decision
        self.opinion_state = np.mean(opinions)

        i=0
        i_max = 1000*self.N
        neighbor = self.N
        while i<i_max:

            #find household with min waiting time
            candidate = self.waiting_times.argmin()

            #remember update_time and incease waiting time of household
            update_time = self.waiting_times[candidate]
            self.waiting_times[candidate] += np.random.exponential(scale=self.tau)

            #load neighborhood of household i
            neighbors = self.neighbors[:,candidate].nonzero()[0]

            #noise in imitation (exploration of strategies)
            rdn = np.random.uniform() 
            if rdn < self.eps*(1-self.phi):
                self.opinions[candidate] = np.random.choice(self.possible_opinions)
                candidate = -1
                break
            #noise in rewiring (sometimes they make new friends at random..)
            elif rdn > 1.-self.eps * self.phi and len(neighbors)>0:
                unconnected = np.zeros(self.N, dtype=int)
                for i in range(self.N):
                    if i not in neighbors:
                        unconnected[i] = 1
                unconnected = unconnected.nonzero()[0]
                if len(unconnected)>0:
                    old_neighbor = np.random.choice(neighbors)
                    new_neighbor = np.random.choice(unconnected)
                    self.neighbors[candidate, old_neighbor] = self.neighbors[old_neighbor, candidate] = 0
                    self.neighbors[candidate, new_neighbor] = self.neighbors[new_neighbor, candidate] = 1
                candidate = -1
                break



            #if candidate has neighbors, chose one at random.
            if len(neighbors)>0:
                neighbor = np.random.choice(neighbors)

                #check if preferences of candidate and random neighbor differ
                if (opinions[candidate] == opinions[neighbor]):

                    #if candidate and neighbor have same preferences, they
                    #not suitable for update. (RETRY)
                    neighbor = self.N

            if neighbor<self.N:
                #update candidate found (GOD)
                break
            elif self.consensus == True:
                candidate = -1
                update_time = self.t + self.tau
                break
            else:
                i += 1
                if i%self.N == 0:
                    if self.detect_consensus_state(opinions):
                        #no update candiate found because of consensus state (GOD)
                        candidate = -1
                        break
            if i>=i_max:
                #no update candidate and no consensus found (BAD)
                candidate = -2

        return candidate, neighbor, neighbors, update_time

    def update_opinion_formation(self, candidate, neighbor, neighbors):

        same_unconnected = np.zeros(self.N, dtype=int)
        opinion = self.investment_decision

        #adapt or rewire?
        if (self.phi == 1 or (self.phi != 1 and np.random.uniform() < self.phi)):
            #if rewire
            for i in xrange(self.N):
                if (opinion[i] == opinion[candidate] and i not in neighbors and i != candidate):
                    same_unconnected[i] = 1
            same_unconnected = same_unconnected.nonzero()[0]
            if len(same_unconnected)>0:
                new_neighbor = np.random.choice(same_unconnected)
                self.neighbors[candidate, neighbor] = self.neighbors[neighbor, candidate] = 0
                self.neighbors[candidate, new_neighbor] = self.neighbors[new_neighbor, candidate] = 1
        else:
            #if adapt
            #compare fitness
            df = self.fitness(neighbor) - self.fitness(candidate)
            if (np.random.uniform() < .5*(np.tanh(df) + 1)):
                    self.opinions[candidate] = self.opinions[neighbor]
        return 0

    def update_decision_making(self):
        #update decision vector for all
        #households depending on their
        #preferences and the state of the
        #economy
        
        #so far, the decision making is governed by
        #an adaptive voter dynamics. Therefore,
        #there is no differentiation between 
        #opinions and decisions.
        self.investment_decisions = self.opinions
        return 0

    def detect_consensus_state(self, opinions):
        #check if network is split in components with
        #same opinions/preferences
        #returns 1 if consensus state is detected, 
        #returns 0 if NO consensus state is detected.
        
        cc = connected_components(self.neighbors, directed=False)[1]
        self.consensus = all(len(np.unique(opinions[c]))==1
                for c in ((cc==i).nonzero()[0]
                for i in np.unique(cc)))

        if self.consensus and self.consensus_state == -1:
            self.consensus_time = self.t

        if self.consensus:
            self.consensus_state = np.mean(opinions)

        return self.consensus

    def fitness(self, agent):
        return self.income[agent]

    def update_economic_trajectory(self):
        self.trajectory.append([self.t, 
                        self.w,
                        self.r_c, 
                        self.r_d, 
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
                        self.consensus,
                        self.opinion_state])

    def init_economic_trajectory(self):
        self.trajectory.append(['time',
                    'wage',
                    'r_c',  
                    'r_d', 
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
                    'opinion state'])

        dt = [self.t, self.t]
        x0 = np.fromiter(chain.from_iterable([
            list(self.investment_clean), 
            list(self.investment_dirty), 
            [self.P, self.G]]), dtype='float')

        [x0,x1] = odeint(self.economy_dot_leontief, x0, dt)

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

    ## Run Parameters

    N = 100                 # number of households
    p = 0.125                 # link density for household network
    P_start = 1000.         # initial population
    t_max = 100             # max runtime
    tau = 1.               # timescale for social update
    phi = 0.5               # rewiring prob. for social update
    adjacency = nx.adj_matrix(nx.erdos_renyi_graph(N,p)).toarray()
    opinions = np.random.randint(2, size=N)

    # Initialize Model

    model = divestment_core(adjacency, opinions, P_start, tau=tau, phi=phi)

    # Turn on debugging

    model.debug = True

    # Run Model

    model.run(t_max)

    # Print some output

    print connected_components(model.neighbors, directed=False)
    print 'investment decisions:'
    print model.investment_decision
    print 'consensus reached?', model.consensus
    print model.consensus_state
    print 'finish time', model.t
    print 'steps computed', model.steps



    trj = model.trajectory
    headers = trj.pop(0)
    df = pd.DataFrame(trj, columns=headers)
    df = df.set_index('time')
    fig = mp.figure()
    ax1 = fig.add_subplot(231)
    df[['K_d']].plot(ax = ax1)
    #mp.axvline(model.consensus_time)
    ax1.set_yscale('log')
   
    ax2 = fig.add_subplot(232)
    df[['P_d', 'P_c']].plot(ax = ax2)
    mp.axvline(model.consensus_time)
    ax2.set_yscale('log')

    ax3 = fig.add_subplot(233)
    df[['G']].plot(ax = ax3)
    mp.axvline(model.consensus_time)

    ax4 = fig.add_subplot(234)
    df[['opinion state']].plot(ax = ax4)
    mp.axvline(model.consensus_time)
    ax4.set_yscale('log')

    ax5 = fig.add_subplot(235)
    df[['r_d', 'r_c']].plot(ax = ax5)
    mp.axvline(model.consensus_time)
    ax5.set_yscale('log')

    ax6 = fig.add_subplot(236)
    df[['R']].plot(ax = ax6)
    mp.axvline(model.consensus_time)
    ax6.set_yscale('log')

    mp.show()
    print df[['P_c', 'P_d', 'G', 'K_c', 'K_d']]
 
 
