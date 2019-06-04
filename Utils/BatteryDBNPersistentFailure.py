# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:52:16 2019

@author: 13383861
"""

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import math


#%%
##################### Persistence Transition Model #####################
#The persistence transition model is a M*K x M*K  matrix representing a factorial HMM,
#where M is the number of possible battery states
#and K is the number of possible BatteryMeterBroken states. 
#Initially assume the battery states are {0,1,2,3,4,5} and the BatteryMeterBroken states are {0,1}
#The tranisition matrix is then a 10x10 matrix, anaagous to the non-factorial HMM, where:
#Rows 0-5 correspond to the previous battery state from 0-5 and the BatteryMeterBroken state 0 (false)
#Rows 5-10 correspond to the previous battery state from 0-5 and the BatteryMeterBroken state 1 (true)
#Columns 0-5 correspond to the current battery state from 0-5 and the current BatteryMeterBroken state 0 (false)
#Columns 5-10 correspond to the current battery state from 0-5 and the current BatteryMeterBroken state 1 (true)
persistent_battery_state_transition_matrix = np.array([
        #first list corresponds to 0 to 0 transition     Second list corresponds to 0 to 1 transition for
        #for BM Broken (Not broken to not broken)        BM broken (not broken to broken)
        [_*0.999 for _ in [0.6,0.1,0.1,0.1,0.05,0.05]] + [_*0.001 for _ in [0.6,0.1,0.1,0.1,0.05,0.05]],
        [_*0.999 for _ in [0.3,0.5,0.1,0.05,0.03,0.02]] + [_*0.001 for _ in [0.3,0.5,0.1,0.05,0.03,0.02]],        
        [_*0.999 for _ in [0.1,0.2,0.5,0.1,0.05,0.05]] + [_*0.001 for _ in [0.1,0.2,0.5,0.1,0.05,0.05]],
        [_*0.999 for _ in [0.05,0.05,0.2,0.5,0.1,0.1]] + [_*0.001 for _ in [0.05,0.05,0.2,0.5,0.1,0.1]],
        [_*0.999 for _ in [0.05,0.05,0.1,0.2,0.5,0.1]] + [_*0.001 for _ in [0.05,0.05,0.1,0.2,0.5,0.1]],
        [_*0.999 for _ in [0.05,0.05,0.1,0.1,0.1,0.6]] + [_*0.001 for _ in [0.05,0.05,0.1,0.1,0.1,0.6]],
        #first list corresponds to 1 to 0 transition     Second list corresponds to 0 to 1 transition for
        #for BM Broken (Broken to not Broken)            (BM Broken to BM Broken)
        [0]*6 + [0.6,0.1,0.1,0.1,0.05,0.05],
        [0]*6 + [0.3,0.5,0.1,0.05,0.03,0.02],
        [0]*6 + [0.1,0.2,0.5,0.1,0.05,0.05],
        [0]*6 + [0.05,0.05,0.2,0.5,0.1,0.1],
        [0]*6 + [0.05,0.05,0.1,0.2,0.5,0.1],
        [0]*6 + [0.05,0.05,0.1,0.1,0.1,0.6]], dtype = np.float64)

#normalize each row to 1
for row_index, row in enumerate(persistent_battery_state_transition_matrix):
    persistent_battery_state_transition_matrix[row_index] = row/row.sum()
    
assert all([math.isclose(persistent_battery_state_transition_matrix[_].sum(),1, rel_tol = 0.0000001) for _ in range(len(persistent_battery_state_transition_matrix))])
##################### Persistence Transition Model #####################

#%%

##################### Persistence Sensor Model #####################
#When sensor is ok, sensor model for BMeter is identical to the transient failure model; when the sensor is broken, 
#it says BMeter is always 0, regardless of the actual battery charge.
no_battery_levels = 6
battery_meter_levels = [_ for _ in range(no_battery_levels)]
#assertion standard_deviation = 2.5
standard_deviation = 2
#normalized discrete approximation of gaussian with mean battery level, sd standard_deviation. This is the sensor model
battery_meter_gaussians = [multivariate_normal([battery_meter_level], [standard_deviation]).pdf(battery_meter_levels)/multivariate_normal([battery_meter_level], [standard_deviation]).pdf(battery_meter_levels).sum() for battery_meter_level in battery_meter_levels]
#no partial observability!
#battery_meter_gaussians = np.identity(6, dtype = np.float64)
batt_meter_matrix_persistent = np.concatenate(battery_meter_gaussians, axis = 0).reshape(no_battery_levels,no_battery_levels)

def get_sensor_model_probability_matrix_persistent_battery(battery_meter_reading):
    '''
    Returns the sensor model corresponding to p(BatterMeter = batter_meter_reading | BatteryMeterBroken, State).
    Implicitly follows order specified by the belief distribution and transition matrix, whereby 
    0-5 corresponds to BatteryMeterBroken = 0 (Battery meter broken is false)
    6-12 corresponds to BatteryMeterBroken = 1 (Battery meter broken is true).
    Maybe this should be a continuous Gaussian as outlined in #http://www.ee.columbia.edu/~sfchang/course/svia-F03/papers/factorial-HMM-97.pdf Page 4 (Factorial Hidden Markov Models), ZOUBIN GHAHRAMANI 
    '''
    if 0 <= battery_meter_reading < no_battery_levels:    
        BMSensorPersistentBatteryModel = np.zeros((no_battery_levels*2,no_battery_levels*2))
        BMSensorNotBrokenMatrix = np.concatenate(batt_meter_matrix_persistent, axis = 0).reshape(no_battery_levels,no_battery_levels)    
        #concatenate all to form 12x12 matrix
        #upper diagonal corresponds to sensor model for not broken matrix
        upper = BMSensorNotBrokenMatrix[battery_meter_reading]
        #lower diagonal corresponds to sensor model for broken matrix
        lower = np.append(np.array([1]), np.zeros((no_battery_levels-1)))
        np.fill_diagonal(BMSensorPersistentBatteryModel, np.append(upper, lower))
        return BMSensorPersistentBatteryModel
    else:
        raise Exception("Please provide a valid sensor reading,")

##################### Persistence Sensor Model #####################

#%%


##################### Battery State Distribution ##################### 
#marginalize over the BatterySensorBroken related variables
def get_battery_sensor_broken_state_distribution(distribution_vector):
    '''
    By convention, the distribution vector is assumed to take the form: 
        
    BMeter not broken, battery = 0
    BMeter not broken, battery = 1
    BMeter not broken, battery = 2
    .
    .
    BMeter not broken, battery = 5
    BMeter broken, battery = 0
    .
    .
    BMeter broken, battery = 5
    To get the battery state distribution, marginalize over the BMeterBroken variable
    '''    
    return np.array([distribution_vector[:no_battery_levels].sum(), distribution_vector[no_battery_levels:].sum()])

##################### Battery State Distribution #####################

#%%
    

##################### BatterySensorBroken Distribution #####################
#marginalize over the BatterySensorBroken related variables
def get_battery_state_distribution(distribution_vector):
    '''
    By convention, the distribution vector is assumed to take the form: 
        
    BMeter not broken, battery = 0
    BMeter not broken, battery = 1
    BMeter not broken, battery = 2
    .
    .
    BMeter not broken, battery = 5
    BMeter broken, battery = 0
    .
    .
    BMeter broken, battery = 5
    To get the BMeterBroken state distribution, marginalize over the Battery variable
    '''
    return np.array([sum([distribution_vector[_], distribution_vector[no_battery_levels+_]]) for _ in range(no_battery_levels)])


##################### BatterySensorBroken Distribution #####################






############################################################################    
##################### DBN with CPDs and Marginalization ####################
############################################################################
    
##################### PersistentBatteryTransitionModel #####################
#As above, could model this as a HMM but it leads to some complexity (cannot simply state CPDs).
#Try the Bayes Net approach of summing out sensorBroken persistent variable.
#Define everything piecewise first

#convention i(rows) corresponds to bm_broken_t_minus_one, j(cols) corresponds to bm_broken_t
batt_m_transition_matrix = np.array([[0.999, 0.001], [0, 1]], dtype = np.float64)
def bm_broken_transition_probability(bm_broken_t, bm_broken_t_minus_one):
    '''
    bm_broken_t takes value 0 if battery meter is not broken, 1 if broken
    '''
    return batt_m_transition_matrix[bm_broken_t_minus_one][bm_broken_t]

def batt_cap_transition_probability(battery_cap_t, battery_cap_t_minus_one):
    '''
    '''
    return battery_state_transition_matrix[battery_cap_t_minus_one][battery_cap_t]

def get_persistent_battery_failure_updated_state_esimate(battery_cap_t, bm_broken_t, b_meter_measurement, previous_probability_vector):
    '''
    Returns the updated state estimate for p(bat_cap_t, bm_broken_t | b_meter_measurment)
    '''
    #print("sum Probability of battery_cap_t = {} over all previous = {}".format(battery_cap_t, sum([batt_cap_transition_probability(battery_cap_t, battery_cap_t_minus_one) for battery_cap_t_minus_one in range(6)])), end = '*')
    #print("Probability of bm_broken_t = {} over all previous = {}".format(bm_broken_t, sum([bm_broken_transition_probability(bm_broken_t, bm_broken_t_minus_one) for bm_broken_t_minus_one in [0,1]])), sum([batt_cap_transition_probability(battery_cap_t, battery_cap_t_minus_one) for battery_cap_t_minus_one in range(6)]), end = '\n\n')
    
    #transition_prob = sum([bm_broken_transition_probability(bm_broken_t, bm_broken_t_minus_one) for bm_broken_t_minus_one in [0,1]]) * sum([batt_cap_transition_probability(battery_cap_t, battery_cap_t_minus_one) for battery_cap_t_minus_one in range(6)])
    
    transition_prob = 0
    for prev_bm_broken_state in [0,1]:
        for prev_batt_cap_state in range(no_battery_levels):
            transition_prob += bm_broken_transition_probability(bm_broken_t, prev_bm_broken_state)* batt_cap_transition_probability(battery_cap_t, prev_batt_cap_state) * previous_probability_vector[prev_bm_broken_state][prev_batt_cap_state]
#            print("bm_broken transition prob: ", bm_broken_transition_probability(bm_broken_t, prev_bm_broken_state))
#            print("b_cap transition prob: ", batt_cap_transition_probability(battery_cap_t, prev_batt_cap_state))
#            print("Prev_prob: ", previous_probability_vector[prev_bm_broken_state][prev_batt_cap_state])
    print("Transition prob to battery_cap_t = {}, bm_broken_t = {} is {}".format(battery_cap_t, bm_broken_t, transition_prob))
    sensor_prob = get_persistent_battery_failure_sensor_model_probability(battery_cap_t, bm_broken_t, b_meter_measurement) 
    #don't forget to normalize once these values have been calculated for joint conditional dist.
    #print("New probability for p(bat_cap_t, bm_broken_t | b_meter_measurment): ".format())
    #print("Transition probabilities sensor_prob * transition_prob * previous_probability: {} * {} * {} = {}".format(sensor_prob, transition_prob, previous_probability, sensor_prob * transition_prob * previous_probability))
    return sensor_prob * transition_prob 
    
def get_persistent_battery_failure_sensor_model_probability(battery_cap_t, bm_broken_t, b_meter_measurement):
    if bm_broken_t == 0:
        #AI:AMA p594. "when sensor is OK, the sensor model for BMeter is identical to the transient failure model"
        return batt_meter_matrix[b_meter_measurement][battery_cap_t]
    else:
        #AI:AMA p594. "when the sensor is broken, it says BMeter is always 0, regardless of actual battery charge"
        return 1 if b_meter_measurement == 0 else 0 #i.e. treat battery_cap_t as the true probability

def update_all_probs(prev_distribution: np.array, b_meter_measurement):
    #assuming prev_distribution is a 2 x 6 vector, where positions (0, 0-5) represent the distribution 
    #of the battery capacity states given that the sensor isn't broken and positions (1, 0-5) represent
    #the distribution of the battery capacity states given that the sensor is broken 
    updated_belief_vector = np.zeros((2, 6), dtype = np.float64)
    #sensor_working_dist = batt_cap_transition_probability[0]
    for sensor_working_value in [0,1]:
        for bat_cap_t_index in range(no_battery_levels):
            updated_belief_vector[sensor_working_value][bat_cap_t_index] = get_persistent_battery_failure_updated_state_esimate(bat_cap_t_index, sensor_working_value , b_meter_measurement, prev_distribution)
    #return normalized beliefs
    return normalize_belief_vector(updated_belief_vector)
    
def get_expected_battery_cap(belief_vector):
    '''
    belief_vector is a 2x6 vector, where first row corresponds to battery_meter not broken, 
    second row corresponds to battery_meter broken. Returns the expected battery capacity
    '''
    return get_expected_value(get_battery_cap_conditional_dist(belief_vector))

def get_battery_cap_conditional_dist(belief_vector):
    '''
    belief_vector is a 2x6 vector, where first row corresponds to battery_meter not broken, 
    second row corresponds to battery_meter broken. Returns the conditional distribution of the battery capacity 
    given the battery meter readings to date
    '''
    return np.sum(belief_vector, axis = 0)

def get_battery_meter_broken_conditional_dist(belief_vector):
    '''
    belief_vector is a 2x6 vector, where first row corresponds to battery_meter not broken, 
    second row corresponds to battery_meter broken.. Returns the conditional distribution of the battery meter broken 
    given the battery meter readings to date.
    '''
    return np.sum(belief_vector, axis = 1)

    
##################### PersistentBatteryTransitionModel #####################

#

############################################################################    
##################### DBN with CPDs and Marginalization ####################
############################################################################


def get_battery_expected_value(belief_state):
    #for i in range(no_battery_levels):
    #    print(belief_state[i]+belief_state[i+no_battery_levels])
    #print([i*(belief_state[i]+belief_state[i+no_battery_levels]) for i in range(len(belief_state)-1)])
    return sum([i*(belief_state[i]+belief_state[i+no_battery_levels]) for i in range(no_battery_levels)])

def get_expected_value(belief_state):
    #print(belief_state)
    return sum([i*belief_state[i] for i in range(len(belief_state))])

#%%
#Test model with persistent failure variable
#by convention, the state distribution vector corresponds to 
# p(battery = 0, batterySensorBroken = False), 
#.
#.
#p(battery = 5, batterySensorBroken = False), 
#p(battery = 0, batterySensorBroken = True), 
#.
#.
#p(battery = 5, batterySensorBroken = True) 

#easier to state values relative to 1 and then normalize
initial_distribution_persistent_battery_sensor_broken = np.array([0.05, 0.1, 0.05, 0.2, 0.05, 0.55,
                                                                  0, 0, 0, 0, 0, 0])
initial_distribution_persistent_battery_sensor_broken = normalize_belief_vector(initial_distribution_persistent_battery_sensor_broken)

initial_distribution_persistent_battery_sensor_broken_factored = initial_distribution_persistent_battery_sensor_broken.reshape((2,6))
#%%
expected_values_persistent = [get_expected_battery_cap(initial_distribution_persistent_battery_sensor_broken_factored)]
batt_cap_dist = [get_battery_cap_conditional_dist(initial_distribution_persistent_battery_sensor_broken_factored)]
batt_sensor_broken_dist = [get_battery_meter_broken_conditional_dist(initial_distribution_persistent_battery_sensor_broken_factored)]


next_dist = update_all_probs(initial_distribution_persistent_battery_sensor_broken_factored, 5)
batt_cap_dist.append(get_battery_cap_conditional_dist(next_dist))
batt_sensor_broken_dist.append(get_battery_meter_broken_conditional_dist(next_dist))
expected_values_persistent.append(get_expected_battery_cap(next_dist))

next_dist = update_all_probs(initial_distribution_persistent_battery_sensor_broken_factored, 5)
batt_cap_dist.append(get_battery_cap_conditional_dist(next_dist))
batt_sensor_broken_dist.append(get_battery_meter_broken_conditional_dist(next_dist))
expected_values_persistent.append(get_expected_battery_cap(next_dist))

next_dist = update_all_probs(initial_distribution_persistent_battery_sensor_broken_factored, 5)
batt_cap_dist.append(get_battery_cap_conditional_dist(next_dist))
batt_sensor_broken_dist.append(get_battery_meter_broken_conditional_dist(next_dist))
expected_values_persistent.append(get_expected_battery_cap(next_dist))

next_dist = update_all_probs(initial_distribution_persistent_battery_sensor_broken_factored, 5)
batt_cap_dist.append(get_battery_cap_conditional_dist(next_dist))
batt_sensor_broken_dist.append(get_battery_meter_broken_conditional_dist(next_dist))
expected_values_persistent.append(get_expected_battery_cap(next_dist))

next_dist = update_all_probs(initial_distribution_persistent_battery_sensor_broken_factored, 5)
batt_cap_dist.append(get_battery_cap_conditional_dist(next_dist))
batt_sensor_broken_dist.append(get_battery_meter_broken_conditional_dist(next_dist))
expected_values_persistent.append(get_expected_battery_cap(next_dist))

next_dist = update_all_probs(initial_distribution_persistent_battery_sensor_broken_factored, 5)
batt_cap_dist.append(get_battery_cap_conditional_dist(next_dist))
batt_sensor_broken_dist.append(get_battery_meter_broken_conditional_dist(next_dist))
expected_values_persistent.append(get_expected_battery_cap(next_dist))

next_dist = update_all_probs(initial_distribution_persistent_battery_sensor_broken_factored, 0)
batt_cap_dist.append(get_battery_cap_conditional_dist(next_dist))
batt_sensor_broken_dist.append(get_battery_meter_broken_conditional_dist(next_dist))
expected_values_persistent.append(get_expected_battery_cap(next_dist))

next_dist = update_all_probs(initial_distribution_persistent_battery_sensor_broken_factored, 0)
batt_cap_dist.append(get_battery_cap_conditional_dist(next_dist))
batt_sensor_broken_dist.append(get_battery_meter_broken_conditional_dist(next_dist))
expected_values_persistent.append(get_expected_battery_cap(next_dist))

next_dist = update_all_probs(initial_distribution_persistent_battery_sensor_broken_factored, 5)
batt_cap_dist.append(get_battery_cap_conditional_dist(next_dist))
batt_sensor_broken_dist.append(get_battery_meter_broken_conditional_dist(next_dist))
expected_values_persistent.append(get_expected_battery_cap(next_dist))

next_dist = update_all_probs(initial_distribution_persistent_battery_sensor_broken_factored, 5)
batt_cap_dist.append(get_battery_cap_conditional_dist(next_dist))
batt_sensor_broken_dist.append(get_battery_meter_broken_conditional_dist(next_dist))
expected_values_persistent.append(get_expected_battery_cap(next_dist))

print(expected_values_persistent)
#expected value of 
plt.plot([_ for _ in range(len(expected_values_persistent))], expected_values_persistent, label = "Expected battery capacity")
plt.plot([_ for _ in range(len(batt_sensor_broken_dist))], [_[1] for _ in batt_sensor_broken_dist], label = "Prob of sensor failure")
plt.legend()
#%%
print(batt_sensor_broken_dist)
plt.clf()
plt.plot([_ for _ in range(len(batt_sensor_broken_dist))], batt_sensor_broken_dist)
