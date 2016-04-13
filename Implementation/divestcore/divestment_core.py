

import os
import datetime 
import csv
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import fsolve
from scipy.sparse.csgraph import connected_components
#from data_visualization import plot_economy, plot_network



class divestment_core:

    def __init__(self, adjacency, oppinions, L_start=10., tau=0.05, phi=.5):
        
        ## General Parameters

        self.debug = False              #turn output for debugging on or off
        self.dt = 0.001                 #stepsize for gaussian integration (time between market clearing)
        self.tau = tau                  #mean waiting time between social updates
        self.phi = phi                   #rewiring probability for adaptive voter model

        ## General Variables

        self.t = 0                  # System Time
        self.consensus = False      # 0 for no consensus, 1 consensus
        self.consensus_state = -1   # -1 for no consensus, 1 for clean consensus, 0 for dirty consensus, in between for fragmentation
        self.trajectory = []        # list to save trajectory of output variables

        ## Household parameters

        self.N = adjacency.shape[0]               #number of households
        self.net_birth_rate = .1                    #birth rate for household members
        self.consumption_level = .5                 #percentage of income consumed
        self.relative_value_of_capital = 0.01       #(fitnes) value of capital over consumption

        ## Household variables

        ## Individual
        self.L = L_start                        #members of one household
        self.investment_dirty = np.ones((self.N))    #household investment in clean capital
        self.investment_clean = np.ones((self.N))    #household investment in dirty capital
        self.household_members = np.ones((self.N))   #members of one household
        self.household_members.fill(L_start)
        self.waiting_times = np.zeros((self.N))      
        self.waiting_times = \
                np.random.exponential(scale=self.tau, size=self.N)  #waiting times between rewiring events for each household
        self.neighbors = adjacency              #adjacency matrix between households
        self.investment_decision = oppinions    #investment decision vector
        self.income = np.zeros((self.N))             #household income (for social update)

        ## Aggregated
        self.K_c_sup = 0    #total clean capital (supply)
        self.K_d_sup = 0    #total dirty capital (supply)
        self.L_sup = 0      #total labor (supply)


        ## Sector parameters

        self.delta_c = .01              # Clean capital depreciation rate
        self.delta_d = self.delta_c     # Dirty capital depreciation rate
        self.delta_r_present = 0.001    # Resource harvest efficiency at present stock. Decreases with decreasing stock

        ## Parameters for Cobb Douglas production (preliminary)

        self.C_c = 1. # solow residual for clean sector
        self.C_d = 3. # solow residual for dirty sector

        self.a_c = .4
        self.b_c = 1. - self.a_c

        self.a_d = .4
        self.b_d = .4

        ## Sector variables

        self.L_c = self.N*L_start/2.
        self.L_d = self.N*L_start/2.

        self.K_c = 1.
        self.K_d = 1.

        self.R_c = 1.
        self.R_d = 1.

        self.Y_c = 0.
        self.Y_d = 0.

        ## derived Sector variables

        self.w = 0.
        self.r_c = 0.
        self.r_d = 0.
        self.R_cost = 0

        ## Ecosystem parameters

        self.R_start = 100. 

        ## Ecosystem variables

        self.R_stock = self.R_start

    def run(self, t_max=1000., t_step=0.001):
        """
        run model for t<t_max or until consensus is reached
        
        Parameter
        ---------
        t_max : float
            The maximum time the system is integrated [Default: 100]
            before run() exits. If the model reaches consensus, or is
            unable to find further update candidated, it ends immediately
        t_step : float
            stepsize for time integration of the model e.g. time
            between market clearing.

        Return
        ------
        exit_status : int
            if exit_status == 1: consensus reached
            if exit_status == 0: no consensus reached at t=t_max
            if exit_status ==-1: no consensus & no update candidates found (BAD)
        """
        self.dt = t_step

        self.init_economic_trajectory()

        while self.t<t_max:
            # Update social until consensus is reached
            candidate, neighbor, neighbors, update_time = self.find_update_candidates()
            if candidate==-1: return 1
            if candidate==-2: return -1
            self.update_economy(update_time)
            self.update_oppinion_formation(candidate, neighbor, neighbors)
            #model.update_decision_making()
        return 0

    def delta_r(self, R_stock):
    #dependence of resource harvest
    #cost on remaining resource stock
    #starts at delta_r_present and
    #increases with decreasing stock
    #if stock is depleted, costs are infinite
        if R_stock>0:
            return self.delta_r_present*(self.R_start/self.R_stock)**2
        if R_stock<=0:
            return float('inf')



    def market_clearing_conditions(self, x, args):
    #market clearing conditions for labor (out_1, out_2)
    #and resource uptake (out_3)
        L_c, L_d, R_d = x
        L_tot, K_c, K_d, a_c, a_d, b_c, b_d, R_stock = args
        out_1 = L_tot - L_c - L_d
        out_2 = self.C_c * K_c**a_c * (1-a_c) * L_c**(-a_c) - self.C_d * K_d**a_d * b_d * L_d**(b_d-1) * R_d**(1-a_d-b_d)
        out_3 = self.delta_r(self.R_stock) * R_d**2. - self.C_d * K_d**a_d * L_d**b_d * (1-a_d-b_d) * R_d**(-a_d-b_d)
        return (out_1, out_2, out_3)

    def update_economy(self, update_time):
        if self.debug: print 'update_economy'
        
        while self.t<update_time:

            self.t += self.dt

            #calculate_factor_supply
            self.K_c_sup = np.sum(self.investment_clean)
            self.K_d_sup = np.sum(self.investment_dirty)
            self.L_sup= np.sum(self.household_members)

            #clear_factor_markets
            self.L_c, self.L_d, self.R_d = fsolve(
                    self.market_clearing_conditions, 
                    ([self.L_c, self.L_d, self.R_d]), 
                    args=([self.L_sup, self.K_c_sup, self.K_d_sup, 
                        self.a_c, self.a_d, self.b_c, self.b_d, self.R_stock]))

            if self.debug:
                print 'L_c', 'L_d', 'R_d', 'L_sup'
                print self.L_c, self.L_d, self.R_d, self.L_sup

            #calculate_wages
            self.w = self.K_c_sup**self.a_c * (1-self.a_c) * self.L_c**(-self.a_c)

            if self.debug:
                print 'K_c', 'K_d', 'L_c', 'a_c', 'w'
                print self.K_c_sup, self.K_d_sup, self.L_c, self.a_c, self.w

            #calculate_capital_returns
            self.r_c = self.C_c * self.a_c * self.K_c_sup**(self.a_c-1) * self.L_c**(1-self.a_c)
            self.r_d = self.C_d * self.a_d * self.K_d_sup**(self.a_d-1) * self.L_d**self.b_d * self.R_d**(1-self.a_d-self.b_d)

            #calculate resource cost
            self.R_cost = self.R_d**3*self.delta_r(self.R_stock)

            #calculate_production_output
            self.Y_c = self.C_c * self.K_c_sup**self.a_c * self.L_c**self.b_c
            self.Y_d = self.C_d * self.K_d_sup**self.a_d * self.L_d**self.b_d * self.R_d**(1-self.a_d-self.b_d)

            #distribute_wages_and_returns
            self.income = self.r_c*self.investment_clean + self.r_d*self.investment_dirty + self.w*self.household_members
            self.investment_clean  += (self.investment_decision*(1-self.consumption_level)*self.income - self.investment_clean*self.delta_c)*self.dt
            self.investment_dirty  += (np.logical_not(self.investment_decision)*(1-self.consumption_level)*self.income - self.investment_dirty*self.delta_d)*self.dt


            #grow population
            self.household_members += self.household_members*self.net_birth_rate*self.dt

            #deteriorate resource stock
            self.R_stock -= self.R_d*self.dt

            if self.debug:
                print 'wage=', self.w, 
                'clean capital r=', self.r_c, 
                'dirty capital r=', self.r_d, 
                'K_c=', self.K_c_sup, 
                'K_d=', self.K_d_sup, 
                'R_stock=', self.R_stock
                print self.Y_c - self.w*self.L_c - self.r_c * self.K_c_sup, 
                self.Y_d - self.w*self.L_d - self.r_d*self.K_d_sup - self.delta_r(self.R_stock)*self.R_d**3

        #output economic data
        self.update_economic_trajectory()

        if self.R_stock <= 0:
            return -1
        elif self.R_stock > 0:
            return 0

    def find_update_candidates(self):
        if self.debug: print 'find_update_candidates'

        #For prototyping, use reduced oppinion formation with only
        #investment decision outcomes as oppinion.

        #THIS MUST BE UPDATED IF CUE ORDERS ARE USED AS OPPINIONS
        oppinions = self.investment_decision

        i=0
        i_max = 1000*self.N
        while i<i_max:

            #find household with min waiting
            candidate = self.waiting_times.argmin()

            #remember update_time and incease waiting time of household
            update_time = self.waiting_times[candidate]
            self.waiting_times[candidate] += np.random.exponential(scale=self.tau)

            #chose random neighbor of household i
            neighbors = self.neighbors[:,candidate].nonzero()[0]

            #if candidate has neighbors, chose one at random.
            if len(neighbors)>0:
                neighbor = np.random.choice(neighbors)

                #check if preferences of candidate and random neighbor differ
                if (oppinions[candidate] == oppinions[neighbor]):

                    #if candidate and neighbor have same preferences, they
                    #not suitable for update. (RETRY)
                    neighbor = self.N

            if neighbor<self.N:
                #update candidate found (GOD)
                break
            else:
                i += 1
                if i%self.N == 0:
                    if self.detect_consensus_state(oppinions):
                        #no update candiate found because of consensus state (GOD)
                        candidate = -1
                        break
            if i>=i_max:
                #no update candidate and no consensus found (BAD)
                candidate = -2

        return candidate, neighbor, neighbors, update_time

    def update_oppinion_formation(self, candidate, neighbor, neighbors):
        if self.debug: print 'update_oppinion_formation'

        same_unconnected = np.zeros(self.N, dtype=int)
        oppinion = self.investment_decision

        #adapt or rewire?
        if (self.phi == 1 or (self.phi != 1 and np.random.uniform() < self.phi)):
            #if rewire
            for i in xrange(self.N):
                if (oppinion[i] == oppinion[candidate] and i not in neighbors and i != candidate):
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
                    self.investment_decision[candidate] = self.investment_decision[neighbor]
        return 0

    def update_decision_making(self):
        if self.debug: print 'update_decision_making'
        #update decision vector for all
        #households depending on their
        #preferences and the state of the
        #economy
        return 0

    def detect_consensus_state(self, oppinions):
        if self.debug: print 'detect_consensus_state'
        #check if network is split in components with
        #same oppinions/preferences
        #returns 1 if consensus state is detected, 
        #returns 0 if NO consensus state is detected.
        
        cc = connected_components(self.neighbors, directed=False)[1]
        self.consensus = all(len(np.unique(oppinions[c]))==1
                for c in ((cc==i).nonzero()[0]
                for i in np.unique(cc)))

        if self.consensus:
            self.consensus_state = np.mean(oppinions)

        return self.consensus

    def fitness(self, agent):
        return self.income[agent]*self.consumption_level #+ \
                #self.relative_value_of_capital*(self.investment_clean[agent] + self.investment_dirty[agent])

    def update_economic_trajectory(self):
        self.trajectory.append([self.t, 
                        self.w,
                        self.r_c, 
                        self.r_d, 
                        self.K_c_sup, 
                        self.K_d_sup, 
                        self.L_c, 
                        self.L_d, 
                        self.R_stock,
                        self.Y_c,
                        self.Y_d,
                        self.L_c*self.w,
                        self.L_d*self.w,
                        self.K_c_sup*self.r_c,
                        self.K_d_sup*self.r_d,
                        self.R_cost,
                        self.consensus_state])

    def init_economic_trajectory(self):
        self.trajectory.append(['time',
                    'wage ',
                    'clean capital r ',  
                    'dirty capital r ', 
                    'K_c ', 
                    'K_d ', 
                    'L_c ', 
                    'L_d ', 
                    'R_stock ',
                    'Y_c ',
                    'Y_d ',
                    'L_cost_c ',
                    'L_cost_d ',
                    'K_cost_c ',
                    'K_cost_d ',
                    'R_cost_d ',
                    'consensus '])

if __name__ == '__main__':
    """
    Perform test run and plot some output to check
    functionality
    """
   
    output_location = datetime.datetime.now().strftime("%d_%m_%H-%M-%Ss") + '_output'

    ## Run Parameters

    N = 100                 # number of households
    p = 0.3                 # link density for household network
    L_start = 10            # initial number of household members
    t_max = 100             # max runtime
    stepsize = 0.001        # stepsize for integration of economy
    tau = 0.5               # timescale for social update
    phi = 0.4               # rewiring prob. for social update
    adjacency = nx.adj_matrix(nx.erdos_renyi_graph(N,p)).toarray()
    oppinions = np.random.randint(2, size=N)

    # Initialize Model

    model = divestment_core(adjacency, oppinions, L_start, tau=tau, phi=phi)

    # Run Model

    model.run(t_max, stepsize)

    # Print some output

    print connected_components(model.neighbors, directed=False)
    print model.investment_decision
    print model.consensus
    print model.consensus_state

    # Save network and oppinion data

    np.savetxt(output_location+'_network', model.neighbors, fmt='%d')
    np.savetxt(output_location+'_labels', model.investment_decision, fmt='%d')

    # Save trajectory data
    
    with open(output_location+'_trajectory', 'wb') as output_file:
        wr = csv.writer(output_file, quoting=csv.QUOTE_NONNUMERIC)
        wr.writerows(model.trajectory)
        

    #plot preliminary results 

#   plot_economy(output_location)
#   plot_network(output_location)

