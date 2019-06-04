# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:24:05 2019

@author: 13383861
"""

'''
This file contains the code modelling a system involving a battery. Some tolerance to failure is built-in whereby a sequence of zeros from 
full charge is unlikely due (taken into account by the sensor model).
The battery is assumed to have a hidden state, but does have sensors reporting back on its status. Refer to AIMA p.590 for more info.
This battery model assumes that the agent is moving at a fixed speed per unit timestep and is updated at timesteps spaced at fixed intervals
It seems to be most useful in the case of sensor failure, but also allows for planning into the future based on previous observations


'''

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import math
from hidden_markov import hmm

#%%
##################### Sensor Model #####################
#battery levels are 0-10
no_battery_levels = 11
battery_meter_levels = [_ for _ in range(no_battery_levels)]
#assertion standard_deviation = 2.5
standard_deviation = 2.5
#normalized discrete approximation of gaussian with mean battery level, variance 0.2. This is the sensor model
battery_meter_gaussians = [multivariate_normal([battery_meter_level], [standard_deviation]).pdf(battery_meter_levels)/multivariate_normal([battery_meter_level], [standard_deviation]).pdf(battery_meter_levels).sum() for battery_meter_level in battery_meter_levels]
#battery_meter_gaussians = np.identity(6, dtype = np.float64)
#i,j position denotes the probability of sensor reading indexed by i, battery capacity indexed by j
batt_meter_matrix = np.matrix(np.concatenate(battery_meter_gaussians, axis = 0).reshape(no_battery_levels, no_battery_levels))
#plt.plot(battery_meter_levels, battery_meter_gaussians)
##################### Sensor Model #####################


def get_sensor_matrix():
    pass
#%%
##################### Transition Model #####################
battery_levels = np.array([_ for _ in range(no_battery_levels)])

means = battery_levels
sds = np.array([0.5 for _ in battery_levels])



#hard code battery state transition matrix for now
#put in a file somewhere and read from it if hard-coding
#each row represents current state from 0 to 5
#each column represents previous state from 0 to 5
#i,j th (row, col) entry represents the probability of transition from state i = x_t-1 to state j = x_t
#transpose i,j th (row, col) entry represent the probability of transition from state i = x_t to j = x_t-1
#this is assuming that the agent moves a distance of 1 unit on each timestep
#battery_state_transition_matrix = np.array([[0.9,0.1,0,0,0,0],
#       [0.3,0.6,0.1,0,0,0],
#       [0.05,0.3,0.6,0.05,0,0],
#       [0,0.05,0.3,0.6,0.05,0],
#       [0,0,0.05,0.3,0.6,0.05],
#       [0,0,0,0.1,0.3,0.6]], dtype = np.float64)

#for every unit of distance travelled, this represents the belief state of the battery level
battery_state_transition_matrix_unit_distance_travelled = np.matrix([[1,0,0,0,0,0],
       [0.2,0.8,0,0,0,0],
       [0.05,0.15,0.8,0,0,0],
       [0,0.05,0.15,0.8,0,0],
       [0,0,0.05,0.15,0.8,0],
       [0,0,0,0.05,0.15,0.8]], dtype = np.float64)

#%%
#create a sub-array that denotes the probability of going from battery capacity i to battery capacity i-k
#this can be calibrated to give the rav a longer or shorter life
battery_degradation_vectors = [[1], [0.2, 0.8], [0.05,0.15,0.8]]
#battery_state_transition_matrix = np.zeros((no_battery_levels, no_battery_levels))

#for _ in range(no_battery_levels):
#    if _ < len(battery_degradation_vectors[-1]):
#        battery_state_transition_matrix[_][:_+1] = battery_degradation_vectors[_]
#    else:
#        battery_state_transition_matrix[_][_+1-len(battery_degradation_vectors[-1]):_+1] = battery_degradation_vectors[-1]
        
def get_battery_degradation_transition_matrix(no_battery_levels, battery_degradation_vectors):
    battery_state_transition_matrix = np.zeros((no_battery_levels, no_battery_levels))
    for _ in range(no_battery_levels):
        if _ < len(battery_degradation_vectors[-1]):
            battery_state_transition_matrix[_][:_+1] = battery_degradation_vectors[_]
        else:
            battery_state_transition_matrix[_][_+1-len(battery_degradation_vectors[-1]):_+1] = battery_degradation_vectors[-1]
    return np.matrix(battery_state_transition_matrix)
    
#%%           
#recharge matrix is the opposite of the above
battery_recharge_vectors = [[1], [0.8, 0.2]]
#recharge_state_transition_matrix = np.zeros((no_battery_levels, no_battery_levels))

#for _ in range(no_battery_levels):
#    if _ < len(battery_recharge_vectors[-1]):
#        recharge_state_transition_matrix[_][:_+1] = battery_recharge_vectors[_]
#    else:
#        recharge_state_transition_matrix[_][_+1-len(battery_recharge_vectors[-1]):_+1] = battery_recharge_vectors[-1]
#
#recharge_state_transition_matrix = np.flip(np.flip(recharge_state_transition_matrix, axis = 0), axis = 1)

def get_recharge_state_transition_matrix(no_battery_levels, battery_recharge_vectors):
    recharge_state_transition_matrix = np.zeros((no_battery_levels, no_battery_levels))
    for _ in range(no_battery_levels):
        if _ < len(battery_recharge_vectors[-1]):
            recharge_state_transition_matrix[_][:_+1] = battery_recharge_vectors[_]
        else:
            recharge_state_transition_matrix[_][_+1-len(battery_recharge_vectors[-1]):_+1] = battery_recharge_vectors[-1]
    recharge_state_transition_matrix = np.flip(np.flip(recharge_state_transition_matrix, axis = 0), axis = 1)
    return np.matrix(recharge_state_transition_matrix )

#%%
battery_state_transition_matrix_with_motion = np.matrix([[0.9,0.1,0,0,0,0],
       [0.3,0.6,0.1,0,0,0],
       [0.05,0.3,0.6,0.05,0,0],
       [0,0.05,0.3,0.6,0.05,0],
       [0,0,0.05,0.3,0.6,0.05],
       [0,0,0,0.1,0.3,0.6]], dtype = np.float64)



#battery_state_transition_matrix = np.identity(6, dtype = np.float64)
assert all([math.isclose(get_battery_degradation_transition_matrix(no_battery_levels, battery_degradation_vectors)[_].sum(),1, rel_tol = 0.0000001) for _ in range(len(battery_state_transition_matrix))])
#plt.plot(battery_levels, battery_gaussians)
##################### Transition Model #####################




#%%
def get_sensor_model_probability_matrix(battery_meter_reading):
    if 0 <= battery_meter_reading < no_battery_levels:    
        return_array = np.zeros((no_battery_levels, no_battery_levels))
        #matrix consists of the sensor model for a given battery level along the diagonal and 
        #zeros everywhere else. This is because state is marginalized over previous state, which is then multiplied
        #by the sensor model probability.
        np.fill_diagonal(return_array,batt_meter_matrix[battery_meter_reading])
        #uncomment this to give a deterministic sensor - it will report the battery's true state.
        #return_array[battery_meter_reading][battery_meter_reading] = 1
        return return_array
    else:
        raise Exception("Please provide a valid sensor reading,")

def get_transition_model_probability_matrix():
    return get_battery_degradation_transition_matrix(no_battery_levels, battery_degradation_vectors)

def get_transition_model_probability_matrix_with_action(action):
    if action == 'recharge':
        return get_recharge_state_transition_matrix(no_battery_levels, battery_recharge_vectors)
    elif action == 'move':
        return get_battery_degradation_transition_matrix(no_battery_levels, battery_degradation_vectors)

def normalize_belief_vector(belief_vector):
    return belief_vector/belief_vector.sum()

def get_next_t_step_no_persistent_battery(battery_meter_reading, previous_belief_state):
    next_belief = np.matmul(get_sensor_model_probability_matrix(battery_meter_reading),np.matmul(get_transition_model_probability_matrix().transpose(), previous_belief_state))
    print("next belief unnormalized: ", next_belief)
    return normalize_belief_vector(next_belief)

def get_next_t_step_fn(transition_fn, sensor_model_fn):
    def get_next_t_step(battery_meter_reading, action,  previous_belief_state):
        next_belief = np.matmul(sensor_model_fn(battery_meter_reading),np.matmul(transition_fn(action).transpose(), previous_belief_state))
        print("next belief unnormalized: ", next_belief)
        return normalize_belief_vector(next_belief)
    return get_next_t_step
        
def get_expected_value(belief_state):
    #print(belief_state)
    return sum([i*belief_state[i] for i in range(len(belief_state))])


#%%
class StochasticBatteryHMM:
    '''
    StochasticBatteryHMM maintains and updates the state of the battery capacity based on stochastic sensor readings.
    '''
    def __init__(self, no_battery_levels, transition_model, sensor_matrix, initial_distribution):
        '''
        :param no_battery_levels: the number of different levels the battery can take on
        :param transition_model: A function which takes and action as input and returns the transition matrix with probabilities of future states given previous states
        :param sensor_model: A function which gives the matrix of probabilities of a sensor value given a true state.
        
        This HMM assumes that the observations lie in the same domain as the true state.
        '''
        self.no_battery_levels = no_battery_levels
        self.valid_states = {_ for _ in range(self.no_battery_levels)}
        self.transition_model = transition_model
        self.sensor_matrix = sensor_matrix
        self.initial_distribution = initial_distribution
        #initialized distribution of battery states
        self.current_distribution = self.initial_distribution
        
    def get_sensor_model_probability_matrix(self, observation):
        if self.is_valid_sensor_reading(observation):
            return_array = np.zeros((self.no_battery_levels, self.no_battery_levels))
            #matrix consists of the sensor model for a given battery level along the diagonal and 
            #zeros everywhere else. This is because state is marginalized over previous state, which is then multiplied
            #by the sensor model probability.
            np.fill_diagonal(return_array,self. sensor_matrix[observation])
            #uncomment this to give a deterministic sensor - it will report the battery's true state.
            #return_array[battery_meter_reading][battery_meter_reading] = 1
            return np.matrix(return_array)
        else:
            raise Exception("Please provide a valid sensor reading,")
            
    
    def predict_forward(self, no_timesteps):
        '''
        Returns the predicted probability distribution of the battery capacity 
        no_timesteps into the future given the current evidence.
        '''
        return self.__predict_forward(no_timesteps, self.current_distribution, True)
    
    def __predict_forward(self, no_timesteps, distribution = None, first_recursive_call = False):
        
        if first_recursive_call:
            distribution = self.current_distribution
        #base case
        if no_timesteps == 0:
            return distribution
        else:        
            return self.__predict_forward(no_timesteps - 1, self.normalize_belief_vector(np.matmul(self.transition_model("move").transpose(), distribution)))
        
    
    
    def normalize_belief_vector(self, belief_vector):
        return belief_vector/belief_vector.sum()
        
    def normalize_current_belief_distribution(self):
        self.current_distribution = self.normalize_belief_vector(self.current_distribution) 
        
    def is_valid_sensor_reading(self, observation):
        return 0 <= observation <= self.no_battery_levels
    
    def is_valid_action(self, action):
        return action in ["recharge", "move"]
    
    def update_estimated_state(self, action, observation):
        assert self.is_valid_sensor_reading(observation)
        assert self.is_valid_action(action)
        print(np.matrix(self.current_distribution).shape)
        print(self.transition_model(action).transpose().shape)
        print("Product of transition model and current distribution: ", np.matmul(self.transition_model(action).transpose(), np.matrix(self.current_distribution)))
        print(np.matmul(self.transition_model(action).transpose(), np.matrix(self.current_distribution)).shape)
        self.current_distribution = np.matmul(get_sensor_model_probability_matrix(observation), np.matmul(self.transition_model(action).transpose(), self.current_distribution))
        self.normalize_current_belief_distribution()
        
    def get_estimated_state(self):
        return self.current_distribution
    
    def get_expected_battery_capacity(self):
        return sum([battery_level*self.current_distribution[battery_level] for battery_level in range(self.no_battery_levels)])
    
    def __update_transition_model(self, new_transition_model_matrix):
        '''
        Updates the transition model from a trained model using the Expectation Maximization algorithm
        '''
        def new_transition_model(action):
            if action == 'recharge':
                return self.transition_model('recharge')
            elif action == 'move':
                return new_transition_model_matrix
        self.transition_model = new_transition_model
        
    def __update_sensor_model(self, new_sensor_model_matrix):
        '''
        Updates the sensor model from a trained model using the Expectation Maximization algorithm
        '''
        self.sensor_matrix = new_sensor_model_matrix
        
    def calibrate_sensor_and_transition_matrices(self, observation_sequences: "a list of tuples of observation sequences", update_models = False, debug = False):
        '''Given a list of observation sequences, calibrates the sensor and transition matrices to most likely probabilities
        based on Baum-Welch algorithm. Currently does not give consistent results - using EM should always converge to the same
        distribution parameters for 'reasonable' initial conditions.
        
        Assuming that subsequent readings are sampled at uniformly spaced timesteps of x seconds. This means will have to perform 1.5 updates
        if the RAV moves 1.5 'units'
        '''
        states = [i for i in range(self.no_battery_levels)]
        possible_observations = [i for i in range(self.no_battery_levels )]
        print(np.matrix(self.transition_model("move")).shape)
        print(np.matrix(self.sensor_matrix).shape)
        markov_hmm = hmm(states,possible_observations,np.matrix(self.initial_distribution),self.transition_model("move"),self.sensor_matrix)
        num_iterations = 10000000
        #hard code in quantities of observations as 1
        self.trained_sensor_model,self.trained_transition_model,self.trained_initial_distribution = markov_hmm.train_hmm(observation_sequences, num_iterations, [1 for _ in range(len(observation_sequences))] )
        # e,t,s contain new emission transition and start probabilities
        if debug:
            print("Transition model frobenius error: ", np.linalg.norm(self.trained_transition_model - np.matrix(self.transition_model("move")), 'fro'))
            print("Sensor model frobenius error: ", np.linalg.norm(self.trained_sensor_model - np.matrix(self.sensor_matrix), 'fro'))
            print("Initial distribution l2 error: ", np.linalg.norm(self.trained_initial_distribution - self.initial_distribution, 2))
        if update_models:
            self.__update_transition_model(self.trained_transition_model)
            self.__update_sensor_model(self.trained_sensor_model)
            if debug:
                print("The battery transition model has now been set to {}".format(self.trained_transition_model))
                print("The sensor model has now been set to {}".format(self.trained_sensor_model))
        else:
            if debug:
                print("Using user-specified transition model: {}".format(self.transition_model('move')))
                print("Using user-specified sensor model: {}".format(self.sensor_matrix))
                
class DefaultStochasticBatteryHMM(StochasticBatteryHMM):
    '''
    For ease of implementation, allow the user to simply specify the maximum battery capacity. The battery state can take on discrete integers in the domain
    {0,..., battery_capacity} and sensor readings lie in the same domain.
    '''
    def __init__(self, no_battery_levels):
        
        #use default matrices and vectors calculated above in the file. Eventually move these to a permament file storage location.
        battery_degradation_t_model = get_battery_degradation_transition_matrix(no_battery_levels, battery_degradation_vectors)
        battery_recharge_t_model = get_recharge_state_transition_matrix(no_battery_levels, battery_recharge_vectors)
        
        def get_transition_model_probability_matrix_with_action(action):
            if action == 'recharge':
                return battery_recharge_t_model
            elif action == 'move':
                return battery_degradation_t_model
        
        super().__init__(no_battery_levels, get_transition_model_probability_matrix_with_action, batt_meter_matrix, np.array([0 for i in range(no_battery_levels-1)] + [1]))

    
#%%    
if __name__ == '__main__':

    #%%
    initial_distribution = np.array([0,0,0,0,0,0,0,0,0,0,1], dtype = np.float64)
    batt_hmm = StochasticBatteryHMM(11, get_transition_model_probability_matrix_with_action, batt_meter_matrix, initial_distribution)
    assert batt_hmm.get_expected_battery_capacity() == 10
    batt_hmm.update_estimated_state('move', 10)
    batt_hmm.get_estimated_state()
    batt_hmm.update_estimated_state('move', 10)
    batt_hmm.get_estimated_state()
    batt_readings = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    batt_hmm.calibrate_sensor_and_transition_matrices([batt_readings], update_models=True, debug=True)
    print(batt_hmm.sensor_matrix.shape)
    print(batt_hmm.transition_model('move').shape)
    batt_hmm.predict_forward(50)
#%%
    
    assert math.isclose(np.matmul(get_sensor_model_probability_matrix(5),np.matmul(get_transition_model_probability_matrix().transpose(), initial_distribution))[5], 0.1682885, rel_tol = 0.000001), "Assuming fixed transition matrix and sensor model matrix"
    #%%
    #in order to see what happens in AI:AMA diagram P. 594 (a), set a small s.d. for the sensor model 
    expected_values = []
    get_next_t_step_not_persistent_failure = get_next_t_step_fn(get_transition_model_probability_matrix_with_action, get_sensor_model_probability_matrix)
    print(get_expected_value(initial_distribution))
    next_dist = get_next_t_step_not_persistent_failure(10, 'move', initial_distribution)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(10, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(8, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(9, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(7, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(9, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(5, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(5, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(0, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    
    next_dist = get_next_t_step_not_persistent_failure(0, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(0, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    
    #next_dist = get_next_t_step_not_persistent_failure(0, 'move', next_dist)
    #expected_values.append(get_expected_value(next_dist))
    #next_dist = get_next_t_step_not_persistent_failure(0, 'move', next_dist)
    #expected_values.append(get_expected_value(next_dist))
    #next_dist = get_next_t_step_not_persistent_failure(0, 'move', next_dist)
    #expected_values.append(get_expected_value(next_dist))
    #next_dist = get_next_t_step_not_persistent_failure(0, 'move', next_dist)
    #expected_values.append(get_expected_value(next_dist))
    #next_dist = get_next_t_step_not_persistent_failure(0, 'move', next_dist)
    #expected_values.append(get_expected_value(next_dist))
    #next_dist = get_next_t_step_not_persistent_failure(0, 'move', next_dist)
    #expected_values.append(get_expected_value(next_dist))
    #next_dist = get_next_t_step_not_persistent_failure(0, 'move', next_dist)
    #expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(5, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(5, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(5, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(5, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(5, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(5, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(3, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(2, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(2, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(2, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(2, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(2, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(2, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(3, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(3, 'recharge', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(4, 'recharge', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(6, 'recharge', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(8, 'recharge', next_dist)
    expected_values.append(get_expected_value(next_dist))
    next_dist = get_next_t_step_not_persistent_failure(3, 'move', next_dist)
    expected_values.append(get_expected_value(next_dist))
    
    print(expected_values)
    plt.plot([_ for _ in range(len(expected_values))], expected_values)
    plt.ylim(0,10)
    
    
    
    
    #%%
    ##################### Train HMM using Baum-Welch #####################
    from hidden_markov import hmm
    states = [i for i in range(no_battery_levels )]
    possible_observation = [i for i in range(no_battery_levels )]
    # Numpy arrays of the data
    start_probability = np.matrix(initial_distribution )
    transition_probability = np.matrix(battery_state_transition_matrix)
    sensor_probability = np.matrix(batt_meter_matrix)
    # Initialize class object
    test = hmm(states,possible_observation,start_probability,transition_probability,sensor_probability)
    dir(test)
    test.states
    test.state_map
    test.start_prob
    
    observations = [(10,9,9,8,7,6,5,5,3,4,3,2,2,1,0,0,0,0)]
    quantities_of_observations = [1]
    num_iterations=1000
    e,t,s = test.train_hmm(observations,num_iterations,quantities_of_observations )
    # e,t,s contain new emission transition and start probabilities
    
    print(t - transition_probability)
    print(e - sensor_probability)
    print(np.linalg.norm(t - transition_probability, 'fro'))
    print(np.linalg.norm(e - sensor_probability, 'fro'))
    
    dir(test)
    help(test.viterbi)
    help(test.forward_algo)
    test.forward_algo((10, 10, 9, 9))
    help(test.log_prob)   
    #%%
    
    ##################### Train HMM using Baum-Welch #####################
    
    
    
    
    
    
    
    
    
    
    
    
    states = ('s', 't')
    possible_observation = ('A','B' )
    # Numpy arrays of the data
    start_probability = np.matrix( '0.5 0.5 ')
    transition_probability = np.matrix('0.6 0.4 ;  0.3 0.7 ')
    emission_probability = np.matrix( '0.3 0.7 ; 0.4 0.6 ' )
    # Initialize class object
    test = hmm(states,possible_observation,start_probability,transition_probability,emission_probability)
    
    observations = ('A', 'B','B','A')
    obs4 = ('B', 'A','B')
    observation_tuple = []
    observation_tuple.extend( [observations,obs4] )
    quantities_observations = [10, 20]
    num_iter=1000
    e,t,s = test.train_hmm(observation_tuple,num_iter,quantities_observations)
    # e,t,s contain new emission transition and start probabilities







