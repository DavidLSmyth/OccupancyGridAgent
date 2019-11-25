# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:29:29 2019

@author: 13383861
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:28:17 2019

@author: 13383861
"""

# This file contains the code to run an agent which has a noisy sensor which searches for multiple sources of 
# evidence 
# For each of k possible sources:
#   Let b represent the belief that a single source is present
#   repeat sample-update until single-source terminates
#   record source location if present else terminate
#   if k loop not yet run out:
#       redistribute prob. mass of location, normalize





import random
import argparse
import sys
import time
import webbrowser
import csv
import os


sys.path.append('.')

from OccupancyGrid.Agents.DerivedOccupancyGridAgent import MultipleSourceDetectingGridAgent, SimpleGridAgent

from Utils.UE4Grid import UE4Grid
from Utils.Vector3r import Vector3r
from Utils.ActionSelection import EpsilonGreedyActionSelection, TSPActionSelection, TSPNNActionSelection, GreedyActionSelection, SaccadicActionSelection, RandomActionSelection
from Analysis.BasicAgentAnalysis import SimpleAgentAnalyser, _generate_agent_heat_map_from_bel_map
from Utils.SensorSimulators import FalsePosFalseNegBinarySensorSimulator, BinarySensorParameters
from Utils.Prior import generate_gaussian_prior, generate_uniform_prior, generate_and_save_prior_fig
from Utils.ProgressBar import progress_bar
from Utils.BeliefMap import MultipleSourceBinaryBeliefMap, create_multiple_source_binary_belief_map
from Utils.BeliefMapVector import BeliefVector

from Config.ConfigParsers import (get_simulated_sensor_from_config, get_grid_from_config,
                                  get_agent_start_pos_from_config, 
                                  get_source_locations_from_config,
                                  get_max_simulation_steps_from_config,
                                  get_sensor_model_params_from_config,
                                  get_SPRT_params_from_config)

from Utils.SearchTermination import SequentialProbRatioTest

import matplotlib.pyplot as plt
#%%




def generate_new_source_pos():
    return random.randint(0, 9), random.randint(0, 9)
    
    
    

def main(subdir, subsubdir, prior, search_strategy, initial_belief_present, sensor_model_fpr, sensor_model_fnr, run_no):
    random.seed(run_no)#args = parse_args(sys.argv[1:])
    #agent_name = args.agent_name
    #print('args: ',args) 
    #%%
    t1 = time.time()
    agent1_name = 'agent1'
    #agent2_name = 'agent2'
#    agent3_name = 'agent3'
    no_runs = 5
    print("Running with following params: ")
    print("Subdir: ", subdir)
    print("Subsubdir: ", subsubdir)
    print("Prior: ", prior)
    print("search_strat: ", search_strategy)
    print("sensor_model_fpr: ", sensor_model_fpr)
    print("sensor_model_fnr: ", sensor_model_fnr)
    print("initial_belief_present: ", initial_belief_present)
    print("no_runs: ", no_runs)
    #%%    
    #x then y
    grid = get_grid_from_config()
    #print(grid.get_grid_points())
    #means = [18,16]
    #uniform_initial = generate_uniform_prior(grid)
    
    source_locations = get_source_locations_from_config()
    assert all([source_location in grid.get_grid_points() for source_location in source_locations])
    assert len(source_locations) > 1
    agent1_start_pos = get_agent_start_pos_from_config(1)
    
    covariance_matrix = [[3.0, 0],[0, 3.0]]
    #gaussian_initial = generate_gaussian_prior(grid, [source_locations[0].x_val, source_locations[0].y_val], covariance_matrix, initial_belief_sum = 0.5)
    
    #initial_belief_present = 0.25
    if prior == 'Gaussian':
        initial_belief = generate_gaussian_prior(grid, [source_locations[0].x_val, source_locations[0].y_val], covariance_matrix, initial_belief_sum = initial_belief_present)
    elif prior == 'Uniform':
        initial_belief = generate_uniform_prior(grid, initial_belief_present)
    else:
        raise Exception("Unknown Prior: {}".format(prior))

    saccadic_selection_method = SaccadicActionSelection(grid)
    nearest_neighbor_selection = GreedyActionSelection(eff_radius = min([grid.lat_spacing, grid.lng_spacing]))
    random_action_selection = RandomActionSelection(grid)
    if len(grid.get_grid_points()) < 200:
        #this takes a long time to initialize if too many grid points are present
        sweep_action_selection_method = TSPActionSelection(grid, agent1_start_pos)
    epsilon_greedy_action_selection_method  = EpsilonGreedyActionSelection(0.2, eff_radius = 4 * min([grid.lat_spacing, grid.lng_spacing]))

    if search_strategy == 'Saccadic':
        ss = saccadic_selection_method
    elif search_strategy == 'NN':
        ss = nearest_neighbor_selection
        
    elif search_strategy == 'EpsilonGreedy':
        ss = epsilon_greedy_action_selection_method
    elif search_strategy == 'Sweep':
        ss = sweep_action_selection_method
    elif search_strategy == 'Random':
        ss = random_action_selection
    else:
        raise Exception("Unknown search strategy - choices are Saccadic, NN, EpsilongGreedy, Sweep, Random")    
    
    
    agent1_simulated_sensor = get_simulated_sensor_from_config(1)
    agent1_simulated_sensor.source_locations = source_locations

    #agent1_sensor_model_fpr, agent1_sensor_model_fnr = get_sensor_model_params_from_config(1).false_positive_rate, get_sensor_model_params_from_config(1).false_negative_rate
    agent1_sensor_model_fpr, agent1_sensor_model_fnr = sensor_model_fpr, sensor_model_fnr
    
    
    #    grid, initial_pos, move_from_bel_map_callable, height, agent_name, occupancy_sensor_simulator, belief_map_class, search_terminator, other_active_agents = [], prior = {}, comms_radius = 1000, logged = True)
    #estimated_state_map = create_multiple_source_binary_belief_map(grid, uniform_initial, agent1_sensor_model_fpr, agent1_sensor_model_fnr)
    
    
    agent1_initial_belief_map = MultipleSourceBinaryBeliefMap(grid, initial_belief, agent1_sensor_model_fpr, agent1_sensor_model_fnr)
    #prior_belief_present, probability_of_falsely_rejecting_source_is_present_given_source_is_present:"p(type 1 error)", probability_of_falsely_accepting_source_is_present_given_source_is_not_present:"p(type 2 error)"
    #agent1_initial_belief_map.get_probability_source_in_grid()
    #agent1_initial_belief_map.current_belief_vector.get_estimated_state()
    search_terminator = SequentialProbRatioTest(0.5, *get_SPRT_params_from_config(1))
    agent1 = SimpleGridAgent(grid, agent1_start_pos, ss.get_move, -10, agent1_name, agent1_simulated_sensor, MultipleSourceBinaryBeliefMap, agent1_initial_belief_map, search_terminator, other_active_agents = [], comms_radius = 2, logged=False)
    #%%
    #results_file = "D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults/MultipleTarget/{}/{}/MultipleTargetProcess{}.csv".format(str(no_sources), search_strategy, run_no)
    covariance_matrix = [[3.0, 0], [0, 3.0]]
    #with open(results_file, 'w') as csv_file:
    #    csv_file.write("TTD\tConcludedLocation\tActualLocation\n")
        #randomize start position
        #randomize target location
    #    write_str = ''
    for run_number in range(no_runs):
        #agent2 = MultipleSourceDetectingGridAgent(grid, agent1_start_pos, epsilon_greedy_action_selection_method.get_move, -10, agent1_name, agent1_simulated_sensor, MultipleSourceBinaryBeliefMap, agent1_initial_belief_map, search_terminator, other_active_agents = [], comms_radius = 2, logged=False)
        #t1 = time.time()
        belief_values = []
        new_agent_start_pos = Vector3r(random.randint(0, 9) , random.randint(0, 9))
        new_source_pos_x = random.randint(0, 9)
        new_source_pos_y = random.randint(0, 9)
        new_source_pos = Vector3r(new_source_pos_x, new_source_pos_y)
        agent1.current_pos_intended = new_agent_start_pos
        agent1_simulated_sensor.source_locations = [new_source_pos]
        #agent1.current_belief_map = MultipleSourceBinaryBeliefMap(grid, uniform_initial, agent1_sensor_model_fpr, agent1_sensor_model_fnr)
        if prior == 'Gaussian':
            agent1.current_belief_map.current_belief_vector.estimated_state = generate_gaussian_prior(grid, [new_source_pos_x, new_source_pos_y], covariance_matrix, initial_belief_sum = initial_belief_present)
        elif prior == 'Uniform': 
            agent1.current_belief_map.current_belief_vector.estimated_state = initial_belief
            
        else:
            raise Exception("Invalid prior belief")
            
        agent1.timestep = 1
        #agent1.occupancy_sensor_simulator = agent1_simulated_sensor
        #print(agent1.current_belief_map.current_belief_vector.get_estimated_state())

        while not agent1.search_terminator.should_end_search(agent1.current_belief_map):
            agent1.iterate_next_timestep()
            belief_values.append(agent1.current_belief_map.current_belief_vector.get_prob_in_grid())
        located_source = agent1.current_belief_map.get_ith_most_likely_component(1)
        #agent1.current_belief_map.save_visualisation
        no_timesteps = agent1.timestep
        plt.clf()
        plt.plot([i for i in range(len(belief_values))], belief_values)
        plt.ylim(0,1)
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative belief target is present")
        plt.savefig("D:\OccupancyGrid\OccupancyGrid\Runners\ThesisResults\{}\{}\{}\BeliefEvolution{}.png".format(subdir, subsubdir,search_strategy,run_number))
        
        
    t2 = time.time()
    print("\n{} runs took {} seconds.".format(no_runs, t2 - t1))
    
if __name__ == '__main__':
    base_dir = 'D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults'
    #os.chdir(base_dir)
    #subdirs = list(os.walk(base_dir))[0][1]
    subdirs = ["Example1", "MiscalibratedSensor", "VaryingInitialBelief"]
    search_strategies = ['Random', 'Saccadic', 'EpsilonGreedy', 'Sweep']
    #designed to run in parallel

    default_prior = "Uniform"
    default_sensor_model_fpr = 0.2
    default_sensor_model_fnr = 0.15
    default_initial_belief_present = 0.5    
    run_no = 0
    for subdir in subdirs:
        for variable_param in list(os.walk(base_dir+'/'+subdir))[0][1]:
            for search_strategy in search_strategies:
                if subdir == "Example1":
                    default_prior = "Uniform"
                    default_sensor_model_fpr = 0.2
                    default_sensor_model_fnr = 0.15
                    default_initial_belief_present = 0.5
                    default_prior = variable_param
                    
                elif subdir == "MiscalibratedSensor":
                    default_prior = "Uniform"
                    default_sensor_model_fpr = 0.2
                    default_sensor_model_fnr = 0.15
                    default_initial_belief_present = 0.5
                    default_sensor_model_fpr = float("0."+variable_param.split('-')[0])
                    default_sensor_model_fnr = float("0."+variable_param.split('-')[1])
                elif subdir == "VaryingInitialBelief":
                    default_prior = "Uniform"
                    default_sensor_model_fpr = 0.2
                    default_sensor_model_fnr = 0.15
                    default_initial_belief_present = 0.5
                    default_initial_belief_present = float("0."+variable_param)
                else:
                    raise Exception("Cannot run with subdir: {}".format(subdir))
                main(subdir, variable_param, default_prior, search_strategy, default_initial_belief_present, default_sensor_model_fpr, default_sensor_model_fnr, run_no)
                run_no +=1