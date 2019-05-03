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






import argparse
import sys
import time
import webbrowser

sys.path.append('.')

from OccupancyGrid.OccupancyGridAgent import OccupancyGridAgent, SimpleGridAgent, SimpleGridAgentWithSources
from Utils.UE4Grid import UE4Grid
from Utils.Vector3r import Vector3r
from Utils.ActionSelection import EpsilonGreedyActionSelection, TSPActionSelection, TSPNNActionSelection, GreedyActionSelection, SaccadicActionSelection
from Analysis.BasicAgentAnalysis import SimpleAgentAnalyser
from Utils.Sensors import SingleSourceSensor, MultipleSourceSensor
from Utils.BeliefMap import BeliefMapComponent, ChungBurdickBeliefMap
from Utils.Prior import generate_gaussian_prior, save_gaussian_prior, generate_uniform_prior
from Utils.ProgressBar import progress_bar









if __name__ == '__main__':
    #args = parse_args(sys.argv[1:])
    #agent_name = args.agent_name
    #print('args: ',args) 
    t1 = time.time()
    agent1_name = 'agent1'
    agent2_name = 'agent2'
    agent3_name = 'agent3'
    
    
    #x then y
    grid = UE4Grid(1, 1, Vector3r(0,0), 10, 8)
    means = [18,16]
    covariance_matrix = [[7.0, 0], [0, 3]]
    prior = generate_gaussian_prior(grid, means, covariance_matrix, initial_belief_sum = 0.5)
    prior = generate_uniform_prior(grid)
    
    
    source_locations = [Vector3r(1,1), Vector3r(5, 6), Vector3r(10,6)]
    agent_start_pos = Vector3r(10,8)
    saccadic_selection_method = SaccadicActionSelection(grid)
    nearest_neighbor_selection = GreedyActionSelection(eff_radius = min([grid.lat_spacing, grid.lng_spacing]))
    sweep_action_selection_method = TSPActionSelection(grid, agent_start_pos)
    epsilon_greedy_action_selection_method  = EpsilonGreedyActionSelection(0.2, eff_radius = 4 * min([grid.lat_spacing, grid.lng_spacing]))
    
    #alpha is the "false alarm" rate (of false positive)
    false_positive_rate = 0.2
    #beta is the "missed detection" rate (or false negative)
    false_negative_rate = 0.12
    #cb_single_source_sensor = SingleSourceSensor(false_positive_rate, false_negative_rate, source_location)

    cb_single_source_sensor = MultipleSourceSensor(false_positive_rate, false_negative_rate, source_locations)

    #agent_name: str, grid: UE4Grid, belief_map_components: typing.List[BeliefMapComponent], prior: typing.Dict[Vector3r, float], alpha: 'prob of false pos', beta: 'prob of false neg', apply_blur = False):    
    #cb_bel_map = ChungBurdickBeliefMap("agent1", grid, [BeliefMapComponent(prior_i, prior[prior_i]) for prior_i in prior], prior, alpha, beta)
    single_source = False
    #alpha = 0.2
    #beta = 0.1
    
    agent3 = SimpleGridAgentWithSources(grid, agent_start_pos, sweep_action_selection_method.get_move, -10, agent3_name, cb_single_source_sensor, other_active_agents = [], comms_radius = 2, prior = prior, logged=False, single_source=single_source, false_positive_rate=false_positive_rate, false_negative_rate=false_negative_rate)
    #agent3 = SimpleGridAgent(grid, Vector3r(0,0), get_move_from_belief_map_epsilon_greedy, -10, 0.3, agent3_name)
        
    #run_t_timesteps([agent1, agent2], 60)
    threshold = 0.9
    
    #agent3.current_belief_map.save_visualisation("D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3BelMapPrior.png")

    max_timesteps = 1200
    print("Running with max {} timesteps".format(max_timesteps))
    no_timesteps_to_discovery = run_t_timesteps([agent3], max_timesteps, threshold)
    no_timesteps_to_discovery = max_timesteps if not no_timesteps_to_discovery else no_timesteps_to_discovery
    
    print("\n\nSaving visualisations")
    #agent1.current_belief_map.save_visualisation("D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent1BelMap.png")
    #agent2.current_belief_map.save_visualisation("D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent2BelMap.png")
