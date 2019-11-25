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

sys.path.append('.')

from OccupancyGrid.Agents.DerivedOccupancyGridAgent import MultipleSourceDetectingGridAgent, SimpleGridAgent

from Utils.UE4Grid import UE4Grid
from Utils.Vector3r import Vector3r
from Utils.ActionSelection import EpsilonGreedyActionSelection, TSPActionSelection, TSPNNActionSelection, GreedyActionSelection, SaccadicActionSelection
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


#%%


def main():
    random.seed(5)
#args = parse_args(sys.argv[1:])
    #agent_name = args.agent_name
    #print('args: ',args) 
    #%%
    t1 = time.time()
    agent1_name = 'agent1'
    #agent2_name = 'agent2'
#    agent3_name = 'agent3'

    #%%    
    #x then y
    grid = get_grid_from_config()
    #print(grid.get_grid_points())
    means = [18,16]
    covariance_matrix = [[7.0, 0], [0, 3]]
    #gaussian_initial= generate_gaussian_prior(grid, means, covariance_matrix, initial_belief_sum = 0.5)
    uniform_initial = generate_uniform_prior(grid)
    
    source_locations = get_source_locations_from_config()
    assert all([source_location in grid.get_grid_points() for source_location in source_locations])
    agent1_start_pos = get_agent_start_pos_from_config(1)


    saccadic_selection_method = SaccadicActionSelection(grid)
    nearest_neighbor_selection = GreedyActionSelection(eff_radius = min([grid.lat_spacing, grid.lng_spacing]))
    if len(grid.get_grid_points()) < 200:
        #this takes a long time to initialize if too many grid points are present
        sweep_action_selection_method = TSPActionSelection(grid, agent1_start_pos)
    epsilon_greedy_action_selection_method  = EpsilonGreedyActionSelection(0.2, eff_radius = 4 * min([grid.lat_spacing, grid.lng_spacing]))

    
    agent1_simulated_sensor = get_simulated_sensor_from_config(1)

    agent1_sensor_model_fpr, agent1_sensor_model_fnr = get_sensor_model_params_from_config(1).false_positive_rate, get_sensor_model_params_from_config(1).false_negative_rate
    #    grid, initial_pos, move_from_bel_map_callable, height, agent_name, occupancy_sensor_simulator, belief_map_class, search_terminator, other_active_agents = [], prior = {}, comms_radius = 1000, logged = True)
    #estimated_state_map = create_multiple_source_binary_belief_map(grid, uniform_initial, agent1_sensor_model_fpr, agent1_sensor_model_fnr)
    
    
    agent1_initial_belief_map = MultipleSourceBinaryBeliefMap(grid, uniform_initial, agent1_sensor_model_fpr, agent1_sensor_model_fnr)
    #prior_belief_present, probability_of_falsely_rejecting_source_is_present_given_source_is_present:"p(type 1 error)", probability_of_falsely_accepting_source_is_present_given_source_is_not_present:"p(type 2 error)"
    #agent1_initial_belief_map.get_probability_source_in_grid()
    #agent1_initial_belief_map.current_belief_vector.get_estimated_state()
    search_terminator = SequentialProbRatioTest(agent1_initial_belief_map.get_probability_source_in_grid(), *get_SPRT_params_from_config(1))
    agent1 = SimpleGridAgent(grid, agent1_start_pos, epsilon_greedy_action_selection_method.get_move, -10, agent1_name, agent1_simulated_sensor, MultipleSourceBinaryBeliefMap, agent1_initial_belief_map, search_terminator, other_active_agents = [], comms_radius = 2, logged=False)
    #%%
    results_file = "D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults/Example1/Experiment1Process5.csv"
    
    with open(results_file, 'w') as csv_file:
        csv_file.write("TTD\tConcludedLocation\tActualLocation\n")
        #randomize start position
        #randomize target location
        write_str = ''
        no_runs = 625
        for run_number in range(no_runs):
            #agent2 = MultipleSourceDetectingGridAgent(grid, agent1_start_pos, epsilon_greedy_action_selection_method.get_move, -10, agent1_name, agent1_simulated_sensor, MultipleSourceBinaryBeliefMap, agent1_initial_belief_map, search_terminator, other_active_agents = [], comms_radius = 2, logged=False)
            #t1 = time.time()
            new_agent_start_pos = Vector3r(random.randint(0, 9) , random.randint(0, 9))
            new_source_pos = Vector3r(random.randint(0, 9) , random.randint(0, 9))
            agent1.current_pos_intended = new_agent_start_pos
            agent1_simulated_sensor.source_locations = [new_source_pos]
            #agent1.current_belief_map = MultipleSourceBinaryBeliefMap(grid, uniform_initial, agent1_sensor_model_fpr, agent1_sensor_model_fnr)
            agent1.current_belief_map.current_belief_vector.estimated_state = uniform_initial
            #agent1.search_terminator = SequentialProbRatioTest(agent1_initial_belief_map.get_probability_source_in_grid(), *get_SPRT_params_from_config(1))

            agent1.timestep = 1
            #agent1.occupancy_sensor_simulator = agent1_simulated_sensor
            #print(agent1.current_belief_map.current_belief_vector.get_estimated_state())


            located_source = agent1.find_source()
            no_timesteps = agent1.timestep
            write_str+=str(no_timesteps) + '\t' + str(located_source.grid_loc) + '\t' + str(new_source_pos) + '\n'
        csv_file.write(write_str)
        
            
        
    t2 = time.time()
    print("\n{} runs took {} seconds.".format(no_runs, t2 - t1))
        #print("\nAgent 1 has terminated the search after {} timesteps".format(agent1.timestep))
        #print("The sources were at locations {}".format(source_locations))
        #print("The agent detected the following locations: {}".format(located_sources))
        #print("\nAgent1 state: \n", agent1.current_belief_map.current_belief_vector.get_estimated_state())
        #sys.exit(0)

#    
#    print(agent1.current_belief_map.get_belief_map_components())
#    print("\n\nSaving visualisations")
#    agent1.current_belief_map.save_visualisation("D:\\OccupancyGrid\\Visualisations\\Agent1BelMap.png", self.timestep)
#    #agent2.current_belief_map.save_visualisation("D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent2BelMap.png")
#    import matplotlib.pyplot as plt
#    from mpl_toolkits.mplot3d import Axes3D
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    ax.scatter([2], [9], [0], cmap='red')
#    ax.scatter([5], [7], [0], cmap='red')
#    plt.xlim(0, 10)
#    plt.ylim(0,10)
#    ax.set_zlim3d(0,0.5)
#    plt.title("True locations of sources of evidence")    
#    
    
    
    
    
    
if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print("Search took {} seconds".format(t2 - t1))