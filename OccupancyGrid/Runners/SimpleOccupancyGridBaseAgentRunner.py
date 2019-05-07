# Sets up a simple grid to test that agent can move, log, etc.

import argparse
import sys
import time
sys.path.append('.')

from OccupancyGrid.OccupancyGridAgent import OccupancyGridAgent, SimpleGridAgent, SimpleGridAgentWithSources
from Utils.UE4Grid import UE4Grid
#from AirSimInterface.types import Vector3r
from Utils.Vector3r import Vector3r
from Utils.ActionSelection import EpsilonGreedyActionSelection, TSPActionSelection, TSPNNActionSelection, GreedyActionSelection
from Analysis.BasicAgentAnalysis import SimpleAgentAnalyser
from Utils.Sensors import RadModel, RadSensor
from Utils.Prior import generate_uniform_prior, generate_gaussian_prior


def run_timestep(occupancy_grid_agent, other_agent_positions):
    occupancy_grid_agent.explore_timestep(other_agent_positions)
        

def run_t_timesteps(occupancy_grid_agents, no_timesteps, threshold):
    #in future make agent return if threshold is met
    for _ in range(no_timesteps):
        for occupancy_grid_agent in occupancy_grid_agents:
            run_timestep(occupancy_grid_agent, [other_agent.current_pos_intended for other_agent in filter(lambda other_agent: occupancy_grid_agent.agent_name != other_agent.agent_name, occupancy_grid_agents)])
            #print("\nAgent {} is at location {}.".format(occupancy_grid_agent.agent_name, occupancy_grid_agent.current_pos_intended))
            print("T step " + str(_))
            if occupancy_grid_agent.current_reading > threshold:
                return _
        
        
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
    pass
    


if __name__ == "__main__":
    #args = parse_args(sys.argv[1:])
    #agent_name = args.agent_name
    #print('args: ',args) 
    t1 = time.time()
    agent1_name = 'agent1'
    agent2_name = 'agent2'
    agent3_name = 'agent3'
    
    #hard code grid for now
    #x then y
    grid = UE4Grid(10, 15, Vector3r(0,0), 180, 150)
    
    sources_locations = [Vector3r(140,150)]#[Vector3r(25,120),Vector3r(140,15)]
    rad_model = RadModel(sources_locations, grid, 50000)
    #within 8m gives 100% reading
    rad_sensor = RadSensor(rad_model, 10)
    #rad_model.plot_falloff("D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\RadFalloff.png")
    #rad_sensor.plot_falloff("D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\RadSensorFalloff.png")
    
#    agent1 = SimpleGridAgentWithSources(grid, Vector3r(20,0), get_move_from_belief_map_epsilon_greedy, -10, 0.1, agent1_name, [Vector3r(80, 120)], other_active_agents = ['agent2'], comms_radius = 2, prior = {grid_loc:0.3 for grid_loc in grid.get_grid_points()})
 #   agent2 = SimpleGridAgentWithSources(grid, Vector3r(40,135), get_move_from_belief_map_epsilon_greedy, -10, 0.1, agent2_name, [Vector3r(80,120)], other_active_agents = ['agent1'], comms_radius = 2, prior = {grid_loc:0.3 for grid_loc in grid.get_grid_points()})
 
    epsilon = 0.05
        
    agent_start_pos = Vector3r(30,60)
    #selection_method = TSPActionSelection(grid, agent_start_pos)
    #nn_selection_method = TSPNNActionSelection(grid, agent_start_pos)
    epsilon_greedy_selection_method = EpsilonGreedyActionSelection(epsilon, 2 * max([grid.get_lat_spacing(), grid.get_lng_spacing()]))
    #allow the agent to move 2 steps in max direction at any one time
    #greedy_selection_method = GreedyActionSelection(2*max([grid.get_lat_spacing(), grid.get_lng_spacing()]))
    
    agent3 = SimpleGridAgentWithSources(grid, agent_start_pos, epsilon_greedy_selection_method.get_move, -10, agent3_name, rad_sensor, other_active_agents = [], comms_radius = 2, prior = generate_uniform_prior(grid, fixed_value = 0.5), logged=False)
    #agent3 = SimpleGridAgent(grid, Vector3r(0,0), get_move_from_belief_map_epsilon_greedy, -10, 0.3, agent3_name)
        
    #run_t_timesteps([agent1, agent2], 60)
    threshold = 0.95
    
    #agent3.current_belief_map.save_visualisation("D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3BelMapPrior.png")    

    max_timesteps = 200
    no_timesteps_to_discovery = run_t_timesteps([agent3], max_timesteps, threshold)
    no_timesteps_to_discovery = max_timesteps if not no_timesteps_to_discovery else no_timesteps_to_discovery
    
    print("Saving visualisations")
    #agent1.current_belief_map.save_visualisation("D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent1BelMap.png")
    #agent2.current_belief_map.save_visualisation("D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent2BelMap.png")

    print('\n----------------------------------------------------------------\n')
    print("Agent1 most likely coordinate: ",agent3.current_belief_map.get_most_likely_coordinate())
    print("Agent2 most likely coordinate: ",agent3.current_belief_map.get_most_likely_coordinate())
    print('\n----------------------------------------------------------------\n')
    #grid, initial_pos, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, prior = {}
   # OccupancyGridAgent(grid, Vector3r(0,0), get_move_from_belief_map_epsilon_greedy, -12, 0.3, agent_name, other_active_agents).explore_t_timesteps(args.no_timesteps)
    analyser = SimpleAgentAnalyser(agent3)
    print('\n----------------------------------------------------------------\n')
    for key, value in analyser.get_analysis().items():
        print(key, "\t\t.....\t\t", value)
    print('\n----------------------------------------------------------------\n')
    
    
    move_visualisation_fp = "D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3MoveMap.png"
    analyser.save_move_visualisation3d(move_visualisation_fp)
    
    move_visualisation_fp2d = "D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3MoveMap2d.png"
    analyser.save_move_visualisation2d(move_visualisation_fp2d)
    
    bel_map_visualisation_fp = "D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3BelMap.png"
    analyser.save_belief_map_visualisation(bel_map_visualisation_fp)
    #move_animation_fp = "D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3MoveAnimation.mp4"
    #analyser.save_move_animation(move_animation_fp, sources_locations)
    heat_map_fp = "D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3HeatMap.png"
    analyser.save_agent_belief_heat_map_at_timestep(heat_map_fp , agent3.observation_manager, no_timesteps_to_discovery, sources_locations)
    
    bel_map_animation_fp = "D:\\ReinforcementLearning\\DetectSourceAgent\\Visualisations\\Agent3HeatMapAnimation.mp4"
    #analyser.save_agent_belief_map_animation(bel_map_animation_fp, agent3.observation_manager, 50)
    
    #print("Video of first 50 steps rendered")
    #time.sleep(10)
    analyser.save_agent_belief_map_animation_with_sources(bel_map_animation_fp, agent3.observation_manager, no_timesteps_to_discovery, sources_locations)
    #destroy_rav(client, agent_name)
    t2 = time.time()
    print("Time taken for agents to locate source: ", t2-t1)
    

    
    #run_n_times([agent3], 10, 1000, 0.8)












