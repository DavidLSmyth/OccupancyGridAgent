# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:01:54 2019

@author: 13383861
"""

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
import time

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


#%%

def generate_new_source_pos():
    return random.randint(0, 9), random.randint(0, 9)

def main(run_no, prior, search_strategy, initial_belief_present, sensor_model_fpr, sensor_model_fnr, no_sources, no_agents):
    random.seed(run_no)#args = parse_args(sys.argv[1:])
    #agent_name = args.agent_name
    #print('args: ',args) 
    #%%
    t1 = time.time()
    agent1_name = 'agent1'
    agent2_name = 'agent2'
    agent3_name = 'agent3'
    print("Running sim with {} agents".format(no_agents))
    print("Running sim with {} targets".format(no_sources))
    print("Running sim with {} initial belief present".format(initial_belief_present))
    print("Running sim with {} prior".format(prior))
    print("Running sim with {} sensor_model_fpr".format(sensor_model_fpr))
    print("Running sim with {} sensor model fnr".format(sensor_model_fnr))
    print("Running sim with {} search strategy".format(search_strategy))

    inp = input("Start running?")
    while True:
        if inp == 'y':
            break
        elif inp == 'n':
            sys.exit()
        else:
            inp = input("Start Running?")
            
    print('\n\n')
    #%%    
    #x then y
    grid = get_grid_from_config()
    #print(grid.get_grid_points())
    #means = [18,16]
    uniform_initial = generate_uniform_prior(grid)
    
    source_locations = get_source_locations_from_config()
    assert all([source_location in grid.get_grid_points() for source_location in source_locations])
    assert len(source_locations) > 1
    agent1_start_pos = get_agent_start_pos_from_config(1)
    agent2_start_pos = get_agent_start_pos_from_config(1)
    agent3_start_pos = get_agent_start_pos_from_config(1)
    
    covariance_matrix = [[3.0, 0],[0, 3.0]]
    #gaussian_initial = generate_gaussian_prior(grid, [source_locations[0].x_val, source_locations[0].y_val], covariance_matrix, initial_belief_sum = 0.5)
    
    #initial_belief_present = 0.25
    if prior == 'Gaussian':
        initial_belief = generate_gaussian_prior(grid, [source_locations[0].x_val, source_locations[0].y_val], covariance_matrix, initial_belief_sum = initial_belief_present)
    elif prior == 'Uniform':
        initial_belief = generate_uniform_prior(grid, initial_belief_sum = initial_belief_present)
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
    
    agent2_simulated_sensor = get_simulated_sensor_from_config(1)
    agent2_simulated_sensor.source_locations = source_locations
    
    agent3_simulated_sensor = get_simulated_sensor_from_config(1)
    agent3_simulated_sensor.source_locations = source_locations

    #agent1_sensor_model_fpr, agent1_sensor_model_fnr = get_sensor_model_params_from_config(1).false_positive_rate, get_sensor_model_params_from_config(1).false_negative_rate
    agent1_sensor_model_fpr, agent1_sensor_model_fnr = sensor_model_fpr, sensor_model_fnr
    
    
    #    grid, initial_pos, move_from_bel_map_callable, height, agent_name, occupancy_sensor_simulator, belief_map_class, search_terminator, other_active_agents = [], prior = {}, comms_radius = 1000, logged = True)
    #estimated_state_map = create_multiple_source_binary_belief_map(grid, uniform_initial, agent1_sensor_model_fpr, agent1_sensor_model_fnr)
    
    
    agent1_initial_belief_map = MultipleSourceBinaryBeliefMap(grid, initial_belief, agent1_sensor_model_fpr, agent1_sensor_model_fnr)
    agent2_initial_belief_map = MultipleSourceBinaryBeliefMap(grid, initial_belief, agent1_sensor_model_fpr, agent1_sensor_model_fnr)
    agent3_initial_belief_map = MultipleSourceBinaryBeliefMap(grid, initial_belief, agent1_sensor_model_fpr, agent1_sensor_model_fnr)
    #prior_belief_present, probability_of_falsely_rejecting_source_is_present_given_source_is_present:"p(type 1 error)", probability_of_falsely_accepting_source_is_present_given_source_is_not_present:"p(type 2 error)"
    #agent1_initial_belief_map.get_probability_source_in_grid()
    #agent1_initial_belief_map.current_belief_vector.get_estimated_state()
    search_terminator1 = SequentialProbRatioTest(agent1_initial_belief_map.get_probability_source_in_grid(), *get_SPRT_params_from_config(1))
    search_terminator2 = SequentialProbRatioTest(agent2_initial_belief_map.get_probability_source_in_grid(), *get_SPRT_params_from_config(1))
    search_terminator3 = SequentialProbRatioTest(agent3_initial_belief_map.get_probability_source_in_grid(), *get_SPRT_params_from_config(1))
    

    if no_agents == 2:
        agent1 = SimpleGridAgent(grid, agent1_start_pos, ss.get_move, -10, agent1_name, agent1_simulated_sensor, MultipleSourceBinaryBeliefMap, agent1_initial_belief_map, search_terminator1, other_active_agents = ["agent2"], comms_radius = 1000, logged=False)
        agent2 = SimpleGridAgent(grid, agent2_start_pos, ss.get_move, -10, agent2_name, agent2_simulated_sensor, MultipleSourceBinaryBeliefMap, agent2_initial_belief_map, search_terminator2, other_active_agents = ["agent1"], comms_radius = 1000, logged=False)
        print("Running Simulation with 2 agents \n ******************************************\n\n\n\n\n")
        occupancy_grid_agents = [agent1, agent2]
        
    elif no_agents == 3:
        agent1 = SimpleGridAgent(grid, agent1_start_pos, ss.get_move, -10, agent1_name, agent1_simulated_sensor, MultipleSourceBinaryBeliefMap, agent1_initial_belief_map, search_terminator1, other_active_agents = ["agent2", "agent3"], comms_radius = 1000, logged=False)
        agent2 = SimpleGridAgent(grid, agent2_start_pos, ss.get_move, -10, agent2_name, agent2_simulated_sensor, MultipleSourceBinaryBeliefMap, agent2_initial_belief_map, search_terminator2, other_active_agents = ["agent1", "agent3"], comms_radius = 1000, logged=False)
        agent3 = SimpleGridAgent(grid, agent3_start_pos, ss.get_move, -10, agent3_name, agent3_simulated_sensor, MultipleSourceBinaryBeliefMap, agent3_initial_belief_map, search_terminator3, other_active_agents = ["agent1", "agent2"], comms_radius = 1000, logged=False)
        print("Running Simulation with 3 agents \n ******************************************\n\n\n\n\n")
        occupancy_grid_agents = [agent1, agent2, agent3]
    #%%
    results_file1 = "D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults/MultipleAgentsComms/{}/{}/MultipleAgentComms1Process{}.csv".format(str(no_agents), search_strategy, run_no)
    results_file2 = "D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults/MultipleAgentsComms/{}/{}/MultipleAgentComms2Process{}.csv".format(str(no_agents), search_strategy, run_no)
    results_file3 = "D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults/MultipleAgentsComms/{}/{}/MultipleAgentComms3Process{}.csv".format(str(no_agents), search_strategy, run_no)
    
    covariance_matrix = [[3.0, 0], [0, 3.0]]
    with open(results_file1, 'w') as csv_file1, open(results_file2, 'w') as csv_file2, open(results_file3, 'w') as csv_file3:
        csv_file1.write("TTD\tConcludedLocation\tActualLocation\n")
        csv_file2.write("TTD\tConcludedLocation\tActualLocation\n")
        csv_file3.write("TTD\tConcludedLocation\tActualLocation\n")
        #randomize start position
        #randomize target location
        
        no_runs = 5000
        for run_number in range(no_runs):
            #agent2 = MultipleSourceDetectingGridAgent(grid, agent1_start_pos, epsilon_greedy_action_selection_method.get_move, -10, agent1_name, agent1_simulated_sensor, MultipleSourceBinaryBeliefMap, agent1_initial_belief_map, search_terminator, other_active_agents = [], comms_radius = 2, logged=False)
            #t1 = time.time()
            write_str1 = ''
            write_str2 = ''
            write_str3 = ''
            #new_agent1_start_pos = Vector3r(random.randint(0, 9) , random.randint(0, 9))
            #new_agent2_start_pos = Vector3r(random.randint(0, 9) , random.randint(0, 9))
            #new_agent3_start_pos = Vector3r(random.randint(0, 9) , random.randint(0, 9))
            #generate1 new sources positions
            new_source_positions = set([])
            while len(new_source_positions) < no_sources:
                new_source_positions.add(Vector3r(*generate_new_source_pos()))
            
            #new_source_pos = Vector3r(new_source_pos_x, new_source_pos_y)
            for agent in occupancy_grid_agents:
                agent.current_pos_intended = Vector3r(random.randint(0, 9) , random.randint(0, 9))
                agent.timestep = 1
                agent.update_dict = {key:0 for key in agent.update_dict.keys()}
            #agent1.current_pos_intended = new_agent1_start_pos
            #agent2.current_pos_intended = new_agent2_start_pos
            #agent3.current_pos_intended = new_agent3_start_pos
            
            
            agent1_simulated_sensor.source_locations = list(new_source_positions)
            agent2_simulated_sensor.source_locations = list(new_source_positions)
            agent3_simulated_sensor.source_locations = list(new_source_positions)

            #print(agent1_simulated_sensor.source_locations)
            #agent1.current_belief_map = MultipleSourceBinaryBeliefMap(grid, uniform_initial, agent1_sensor_model_fpr, agent1_sensor_model_fnr)
            if prior == 'Gaussian':
                for agent in occupancy_grid_agents:
                    agent.current_belief_map.current_belief_vector.estimated_state = generate_gaussian_prior(grid, [new_source_pos_x, new_source_pos_y], covariance_matrix, initial_belief_sum = initial_belief_present)
            elif prior == 'Uniform': 
                for agent in occupancy_grid_agents:
                    agent.current_belief_map.current_belief_vector.estimated_state = uniform_initial
                
            else:
                raise Exception("Invalid prior belief")
            

            #agent1.occupancy_sensor_simulator = agent1_simulated_sensor
            #print(agent1.current_belief_map.current_belief_vector.get_estimated_state())

            #print("source_location: ", new_source_positions)
            #located_sources = agent1.find_sources(no_sources)
            while not all([agent.search_terminator.should_end_search(agent.current_belief_map) for agent in occupancy_grid_agents]):
                for occupancy_grid_agent in occupancy_grid_agents:
                #print("\nAgent {} is at location {}.".format(occupancy_grid_agent.agent_name, occupancy_grid_agent.current_pos_intended))
                #print("T step " + str(_))
                    if not occupancy_grid_agent.search_terminator.should_end_search(occupancy_grid_agent.current_belief_map):
                        occupancy_grid_agent.iterate_next_timestep()
                        #[other_agent.current_pos_intended for other_agent in filter(lambda other_agent: occupancy_grid_agent.agent_name != other_agent.agent_name, occupancy_grid_agents)])

            located_sources = []
            for agent in occupancy_grid_agents:
                located_sources.append(agent.current_belief_map.get_ith_most_likely_component(1))
                agent.init_observations_file(agent.directories_mapping['observationdir'] + "/{}.csv".format(agent.agent_name))
                
            #located_sources_agent1 = agent1.current_belief_map.get_ith_most_likely_component(1)
            #located_sources_agent2 = agent2.current_belief_map.get_ith_most_likely_component(1)
            #located_sources_agent3 = agent3.current_belief_map.get_ith_most_likely_component(1)
            
            new_source_location = new_source_positions.pop()
            
            write_str1+=str(agent1.timestep) + '\t' + str(located_sources[0].grid_loc) + '\t' + str(new_source_location) + '\n'
            write_str2+=str(agent2.timestep) + '\t' + str(located_sources[1].grid_loc) + '\t' + str(new_source_location)  + '\n'
            if len(occupancy_grid_agents) == 3:
                write_str3+=str(agent3.timestep) + '\t' + str(located_sources[2].grid_loc) + '\t' + str(new_source_location) + '\n'
            if not run_number % 25:
                print("Finished {}".format(run_number))
            csv_file1.write(write_str1)
            csv_file2.write(write_str2)
            if len(occupancy_grid_agents) == 3:
                csv_file3.write(write_str3)
        
        
        
    t2 = time.time()
    print("\n{} runs took {} seconds.".format(no_runs, t2 - t1))
    
if __name__ == '__main__':
#designed to run in parallel
    run_number = sys.argv[8]
    no_agents = sys.argv[7]
    no_sources = sys.argv[6]
    sensor_model_fpr = sys.argv[4]
    sensor_model_fnr = sys.argv[5]
    prior = sys.argv[2]    
    initial_belief_present = sys.argv[3]
    assert prior != ''
    search_strategy = sys.argv[1]
    assert search_strategy != ''
    print("run no: ", run_number)
    t1 = time.time()
    main(int(run_number), prior, search_strategy, float(initial_belief_present), float(sensor_model_fpr), float(sensor_model_fnr), int(no_sources), int(no_agents))
    t2 = time.time()
    print("Search took {} seconds".format(t2 - t1))