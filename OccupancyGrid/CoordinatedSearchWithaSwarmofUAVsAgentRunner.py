# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:24:38 2019

@author: 13383861
"""

# Attempt to recreate results of 
#A Decision-Making Framework for Control Strategies in Probabilistic Search   Timothy H. Chung and Joel W. Burdick

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
from Utils.Sensors import RadModel, RadSensor, ChungBurdickSingleSourceSensor
from Utils.BeliefMap import BeliefMapComponent, ChungBurdickBeliefMap
from Utils.Prior import generate_gaussian_prior, save_gaussian_prior, generate_uniform_prior
from Utils.ProgressBar import progress_bar

#%%

def run_timestep(occupancy_grid_agent, other_agent_positions):
    occupancy_grid_agent.explore_timestep(other_agent_positions)
        

def run_t_timesteps(occupancy_grid_agents, no_timesteps, threshold):
    #in future make agent return if threshold is met
    five_more_steps = 0
    for timestep in range(no_timesteps):
    #timestep = 0
    #while timestep < no_timesteps:
        progress_bar(timestep, no_timesteps-1, status = "Simulation complete" if timestep == no_timesteps-1 else "Simulation in progress")
        for occupancy_grid_agent in occupancy_grid_agents:
            run_timestep(occupancy_grid_agent, [other_agent.current_pos_intended for other_agent in filter(lambda other_agent: occupancy_grid_agent.agent_name != other_agent.agent_name, occupancy_grid_agents)])
            #print("\nAgent {} is at location {}.".format(occupancy_grid_agent.agent_name, occupancy_grid_agent.current_pos_intended))
            #print("T step " + str(_))
            if occupancy_grid_agent.current_belief_map.get_probability_source_in_grid() < threshold:
                five_more_steps = 0
                
            if occupancy_grid_agent.current_belief_map.get_probability_source_in_grid() > threshold and five_more_steps < 4:
                five_more_steps+=1
                
            if five_more_steps >= 4:
                return timestep
            
                #only run for a few more steps
                 #no_timesteps = timestep + 5
                 
        timestep += 1
    return timestep 
        
        
def parse_args(args):
    parser = argparse.ArgumentParser(description='''Run an Occupancy Grid Agent in a UE4 world with the AirSim plugin running''')
    parser.add_argument('no_agents', type=int, help='The number of agents to run in the grid environment', action='store')
    parser.add_argument('no_timesteps', type=int, help='The number of timesteps for the agent to explore', action='store')
    return parser.parse_args(args)


def run_n_times(agents, no_times_to_run, max_timesteps, threshold):
    '''Given a list of agents, runs the search n times and gathers stats'''
    stat_list = {agent.agent_name: [] for agent in agents}
    for _ in range(no_times_to_run):
        run_t_timesteps(agents, max_timesteps, threshold)
        analysers = [SimpleAgentAnalyser(agent) for agent in agents]
        for agent_index, agent in enumerate(agents):
            stat_list[agent.agent_name].append(analysers[agent_index].get_analysis())
            agent.reset()
            
            
            
def run_coordinate_search_with_swarm_UAVs():
    '''Try and replicate results of Coordinated Search with a Swarm of UAVs'''
    #grid = UE4Grid(10, 10, Vector3r(0,0), 90, 90)
    pass


if __name__ == "__main__":
    #args = parse_args(sys.argv[1:])
    #agent_name = args.agent_name
    #print('args: ',args) 
    t1 = time.time()
    agent1_name = 'agent1'
    agent2_name = 'agent2'
    agent3_name = 'agent3'
    
    
    #x then y
    grid = UE4Grid(1, 1, Vector3r(0,0), 10, 10)    
    means = [1,3]
    covariance_matrix = [[7.0, 0], [0, 15]]
    prior = generate_gaussian_prior(grid, means, covariance_matrix, initial_belief_sum = 0.5)
    prior = generate_uniform_prior(grid)
    
    
    source_location = Vector3r(4,3)
    agent_start_pos = Vector3r(5,8)
    saccadic_selection_method = SaccadicActionSelection(grid)
    nearest_neighbor_selection = GreedyActionSelection(eff_radius = min([grid.lat_spacing, grid.lng_spacing]))

    #alpha is the "false alarm" rate (of false positive)
    alpha = 0.2
    #beta is the "missed detection" rate (or false negative)
    beta = 0.25
    cb_single_source_sensor = ChungBurdickSingleSourceSensor(alpha, beta, source_location)

    #agent_name: str, grid: UE4Grid, belief_map_components: typing.List[BeliefMapComponent], prior: typing.Dict[Vector3r, float], alpha: 'prob of false pos', beta: 'prob of false neg', apply_blur = False):    
    #cb_bel_map = ChungBurdickBeliefMap("agent1", grid, [BeliefMapComponent(prior_i, prior[prior_i]) for prior_i in prior], prior, alpha, beta)
    single_source = True
    #alpha = 0.2
    #beta = 0.1
    
    agent3 = SimpleGridAgentWithSources(grid, agent_start_pos, nearest_neighbor_selection.get_move, -10, agent3_name, cb_single_source_sensor, other_active_agents = [], comms_radius = 2, prior = prior, logged=False, single_source=single_source, alpha=alpha, beta=beta)
    #agent3 = SimpleGridAgent(grid, Vector3r(0,0), get_move_from_belief_map_epsilon_greedy, -10, 0.3, agent3_name)
        
    #run_t_timesteps([agent1, agent2], 60)
    threshold = 0.95
    
    #agent3.current_belief_map.save_visualisation("D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3BelMapPrior.png")

    max_timesteps = 500
    no_timesteps_to_discovery = run_t_timesteps([agent3], max_timesteps, threshold)
    no_timesteps_to_discovery = max_timesteps if not no_timesteps_to_discovery else no_timesteps_to_discovery
    
    print("\n\nSaving visualisations")
    #agent1.current_belief_map.save_visualisation("D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent1BelMap.png")
    #agent2.current_belief_map.save_visualisation("D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent2BelMap.png")

    #grid, initial_pos, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, prior = {}
   # OccupancyGridAgent(grid, Vector3r(0,0), get_move_from_belief_map_epsilon_greedy, -12, 0.3, agent_name, other_active_agents).explore_t_timesteps(args.no_timesteps)
    analyser = SimpleAgentAnalyser(agent3)
    print('\n----------------------------------------------------------------\n')
    for key, value in analyser.get_analysis().items():
        print(key, "\t\t.....\t\t", value)
    print('\n----------------------------------------------------------------\n')
    
    
    #move_visualisation_fp = "D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3MoveMap.png"
    #analyser.save_move_visualisation3d(move_visualisation_fp)
    
    move_visualisation_fp2d = "D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3MoveMap2d.png"
    analyser.save_move_visualisation2d(move_visualisation_fp2d)
    
    bel_map_visualisation_fp = "D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3BelMap.png"
    analyser.save_belief_map_visualisation(bel_map_visualisation_fp)
    #move_animation_fp = "D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3MoveAnimation.mp4"
    #analyser.save_move_animation(move_animation_fp, sources_locations)
    heat_map_fp = "D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3HeatMap.png"
    analyser.save_agent_belief_heat_map_at_timestep(heat_map_fp, no_timesteps_to_discovery, source_location)
    
    bel_map_animation_fp = "D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3HeatMapAnimation.mp4"
    #analyser.save_agent_belief_map_animation(bel_map_animation_fp, agent3.observation_manager, 50)
    
    prior_map_visualistaion_fp = "D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3BelMapPrior.png"
    analyser.save_prior_plot(prior_map_visualistaion_fp)
    
    sum_belief_visualistaion_fp = "D://ReinforcementLearning//DetectSourceAgent//Visualisations/Agent3SumBelTime.png"
    analyser.save_sum_belief_until_timestep(no_timesteps_to_discovery, sum_belief_visualistaion_fp)
    #print("Video of first 50 steps rendered")
    #time.sleep(10)
    
    
    analyser.save_agent_belief_map_animation_with_sources(bel_map_animation_fp, no_timesteps_to_discovery, source_location)
    #destroy_rav(client, agent_name)
    t2 = time.time()
    print("Time taken for agents to locate source: ", t2-t1)
    
#    webbrowser.open_new_tab(move_visualisation_fp2d)
#    webbrowser.open_new_tab(move_visualisation_fp2d)
#    webbrowser.open_new_tab(bel_map_visualisation_fp)
#    webbrowser.open_new_tab(heat_map_fp)
    
    #run_n_times([agent3], 10, 1000, 0.8)












