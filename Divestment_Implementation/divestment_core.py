

import numpy as np
from scipy.optimize import fsolve
from scipy.sparse.csgraph import connected_components
import os
import datetime 
from data_visualization import plot_economy, plot_network



class core:

    def __init__(self, N, L_start, integration_step_size):
        
        ## General Parameters

        self.dt = integration_step_size #stepsize for gaussian integration (time between market clearing)
        self.update_timescale = .05      #mean waiting time between social updates
        self.p_rewire = .8              #rewiring probability for adaptive voter model

        ## General Variables

        self.t = 0                  # System Time
        self.consensus = -1         # 0 for no consensus, 1 consensus
        self.consensus_state = -1   # -1 for no consensus, 1 for clean consensus, 0 for dirty consensus, in between for fragmentation

        ## Household parameters

        self.N = N                                  #number of households
        self.net_birth_rate = .1                    #birth rate for household members
        self.consumption_level = .5                 #percentage of income consumed
        self.relative_value_of_capital = 0.01       #(fitnes) value of capital over consumption

        ## Household variables

        ## Individual
        self.L = L_start                        #members of one household
        self.investment_dirty = np.ones((N))    #household investment in clean capital
        self.investment_clean = np.ones((N))    #household investment in dirty capital
        self.household_members = np.ones((N))   #members of one household
        self.household_members.fill(self.L)
        self.waiting_times = np.zeros((N))      #waiting times between rewiring events for each household
        self.neighbors = np.ndarray(dtype=int, shape=(N,N))     #adjacency matrix between households
        self.investment_decision = np.random.randint(2, size=N) #investment decision vector
        self.income = np.zeros((N))                             #household income (for social update)

        ## Aggregated
        self.K_c_sup = 0    #total clean capital (supply)
        self.K_d_sup = 0    #total dirty capital (supply)
        self.L_sup = 0      #total labor (supply)


        ## Sector parameters

        self.delta_c = .5               # Clean capital depreciation rate
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

        self.L_c = 1.
        self.L_d = 1.

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
        
        t_start = self.t
        Delta_t = update_time - t_start
        while self.t<t_start+Delta_t:

            self.t += self.dt

            #calculate_factor_supply
            self.K_c_sup = np.sum(self.investment_clean)
            self.K_d_sup = np.sum(self.investment_dirty)
            self.L_sup= np.sum(self.household_members)

            #clear_factor_markets
            self.L_c, self.L_d, self.R_d = fsolve(
                    self.market_clearing_conditions, 
                    #([self.L_c, self.L_d, self.R_d]), 
                    ([self.L_c, self.L_d, self.R_d]), 
                    args=([self.L_sup, self.K_c_sup, self.K_d_sup, 
                        self.a_c, self.a_d, self.b_c, self.b_d, self.R_stock]))

            if debug:
                print 'L_c', 'L_d', 'R_d'
                print self.L_c, self.L_d, self.R_d, self.L_sup

            #calculate_wages
            self.w = self.K_c_sup**self.a_c * (1-self.a_c) * self.L_c**(-self.a_c)

            if debug:
                print 'K_c', 'K_d', 'L_c', 'a_c', 'w'
                print self.K_c_sup, self.K_d_sup, self.L_c, self.a_c, self.w

            #calculate_capital_returns
            self.r_c = self.C_c * self.a_c * self.K_c_sup**(self.a_c-1) * self.L_c**(1-self.a_c)
            self.r_d = self.C_d * self.a_d * self.K_d_sup**(self.a_d-1) * self.L_d**self.b_d * self.R_d**(1-self.a_d-self.b_d)
            r_resource = self.C_d * self.K_d_sup**self.a_d * self.L_d**self.b_d * (1-self.a_d-self.b_d) * self.R_d**(-self.a_d-self.b_d)

            #calculate resource cost
            self.R_cost = self.R_d**3*self.delta_r(self.R_stock)

            #calculate_production_output
            self.Y_c = self.C_c * self.K_c_sup**self.a_c * self.L_c**self.b_c
            self.Y_d = self.C_d * self.K_d_sup**self.a_d * self.L_d**self.b_d * self.R_d**(1-self.a_d-self.b_d)

            #distribute_wages_and_returns
            self.income = self.r_c*self.investment_clean + self.r_d*self.investment_dirty + self.w*self.household_members
            self.investment_clean  += (self.investment_decision*(1-self.consumption_level)*self.income - self.investment_clean*self.delta_c)*self.dt
            self.investment_dirty  += (np.logical_not(self.investment_decision)*(1-self.consumption_level)*self.income - self.investment_dirty*self.delta_d)*self.dt

            #output economic data
            self.economy_output(output_location)

            #grow population
            self.household_members += self.household_members*self.net_birth_rate*self.dt

            #deteriorate resource stock
            self.R_stock -= self.R_d*self.dt

            if debug:
                print 'wage=', self.w, 
                'clean capital r=', self.r_c, 
                'dirty capital r=', self.r_d, 
                'K_c=', self.K_c_sup, 
                'K_d=', self.K_d_sup, 
                'R_stock=', self.R_stock
                print self.Y_c - self.w*self.L_c - self.r_c * self.K_c_sup, 
                self.Y_d - self.w*self.L_d - self.r_d*self.K_d_sup - self.delta_r(self.R_stock)*self.R_d**3

        if self.R_stock <= 0:
            return -1
        elif self.R_stock > 0:
            return 0

    def economy_output(self, output_location):
            with open(output_location, "a") as myfile:
                myfile.write( '%016.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \t \
                        %16.8g \n' %
                        (self.t, 
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
                        self.consensus_state))

    def init_economy_output(self, output_location):
        with open(output_location, "w") as myfile:
            myfile.write('time \t' +
                    'wage \t ' +
                    'clean capital r \t ' +  
                    'dirty capital r \t ' + 
                    'K_c \t ' + 
                    'K_d \t ' + 
                    'L_c \t ' + 
                    'L_d \t ' + 
                    'R_stock \t ' +
                    'Y_c \t ' +
                    'Y_d \t ' +
                    'L_cost_c \t ' +
                    'L_cost_d \t ' +
                    'K_cost_c \t ' +
                    'K_cost_d \t ' +
                    'R_cost_d \t ' +
                    'consensus \t ' +
                    '\n')
            myfile.write('d_c = ' + `model.delta_c` + 
                    ', d_r = ' + `model.delta_r_present` + 
                    ', a_c = ' + `model.a_c` + 
                    ', b_c = ' + `model.b_c` + 
                    ', a_d = ' + `model.a_d` + 
                    ', b_d = ' + `model.b_d` + '\n')

    def initialize_social(self):

        #initialize network
        a = np.random.randint(2, size=(self.N,self.N))
        a[np.tril_indices(self.N)] = 0
        self.neighbors = a + a.T

        #initialize waiting times
        self.waiting_times = np.random.exponential(scale=self.update_timescale, size=self.N)

        #initialize preferences (structure of individual decision trees)

        #initialize investment decision vector
        self.investment_decision = np.random.randint(2, size=self.N)

        return 0

    def find_update_candidates(self):
        if debug: print 'find_update_candidates'

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
            self.waiting_times[candidate] += np.random.exponential(scale=self.update_timescale)

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

            if neighbor<N:
                #update candidate found (GOD)
                break
            else:
                i += 1
                if i%N == 0:
                    if self.detect_consensus_state(oppinions):
                        #no update candiate found because of consensus state (GOD)
                        candidate = -1
                        print 'case: consensus', candidate
                        break
            if i>=i_max:
                #no update candidate and no consensus found (BAD)
                'case: no update candidate found', candidate
                candidate = -2

        return candidate, neighbor, neighbors, update_time

    def update_oppinion_formation(self, candidate, neighbor, neighbors):
        if debug: print 'update_oppinion_formation'

        same_unconnected = np.zeros(self.N, dtype=int)
        oppinion = self.investment_decision

        #adapt or rewire?
        if (self.p_rewire == 1 or (self.p_rewire != 1 and np.random.uniform() < self.p_rewire)):
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
        #update decision vector for all
        #households depending on their
        #preferences and the state of the
        #economy
        return 0

    def detect_consensus_state(self, oppinions):
        if debug: print 'detect_consensus_state'
        #check if network is split in components with
        #same oppinions/preferences
        #returns 1 if consensus state is detected, 
        #returns 0 if NO consensus state is detected.
        
        cc = connected_components(self.neighbors, directed=False)[1]
        consensus = all(len(np.unique(oppinions[c]))==1
                for c in ((cc==i).nonzero()[0]
                for i in np.unique(cc)))

        if consensus:
            self.consensus_state = np.mean(oppinions)

        return consensus

    def fitness(self, agent):
        return self.income[agent]*self.consumption_level #+ \
                #self.relative_value_of_capital*(self.investment_clean[agent] + self.investment_dirty[agent])


## Run Parameters

N = 50                 # number of households
L_start = 10            # initial number of household members
t_max = 15              # max runtime
stepsize = 0.001        # stepsize for integration of economy

# Initialize Output

debug = False
output_location = datetime.datetime.now().strftime("%d_%m_%H-%M-%Ss") + '_output'


# Initialize Model

model = core(N, L_start, stepsize)
model.initialize_social()
model.init_economy_output(output_location)

model.consensus = False # if False, oppinion formation is on, if True, oppinion formation is off
update_time = 0

# Run Model

while model.t<t_max:
    
    # Update social until consensus is reached
    if not model.consensus:
        candidate, neighbor, neighbors, update_time = model.find_update_candidates()

        print model.t, update_time, model.R_stock
        if candidate>=0:
            model.update_economy(update_time)
            model.update_oppinion_formation(candidate, neighbor, neighbors)
            #model.update_decision_making()
        elif candidate==-1:
            print 'consensus reached!', model.consensus_state
            model.consensus = 1
        elif candidate==-2:
            print 'no update candidates found', candidate
            break

    # After consensus is reached, update economy only until resource is depleted.
    if model.consensus:
        update_time += model.update_timescale
        if model.update_economy(update_time) == -1:
            print 'resource_depleted', model.R_stock

print connected_components(model.neighbors, directed=False)
print model.investment_decision

np.savetxt(output_location+'_network', model.neighbors, fmt='%d')
np.savetxt(output_location+'_labels', model.investment_decision, fmt='%d')


#plot preliminary results 

plot_economy(output_location)
plot_network(output_location)

