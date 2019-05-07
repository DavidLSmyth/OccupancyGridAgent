# Sets up a simple grid to test that agent can move, log, etc.

import argparse
import sys
sys.path.append('.')

from OccupancyGrid.OccupancyGridAgent import OccupancyGridAgent, SimpleGridAgent
from Utils.UE4Grid import UE4Grid
from AirSimInterface.types import Vector3r
from Utils.ActionSelection import get_move_from_belief_map_epsilon_greedy

def run_timestep(occupancy_grid_agent, other_agent_positions):
    occupancy_grid_agent.explore_timestep(other_agent_positions)
        

def run_t_timesteps(occupancy_grid_agents, no_timesteps):
    for timestep in range(no_timesteps):

        for occupancy_grid_agent in occupancy_grid_agents:
            run_timestep(occupancy_grid_agent, [other_agent.current_pos_intended for other_agent in filter(lambda other_agent: occupancy_grid_agent.agent_name != other_agent.agent_name, occupancy_grid_agents)])
        
def parse_args(args):
    parser = argparse.ArgumentParser(description='''Run an Occupancy Grid Agent in a UE4 world with the AirSim plugin running''')
    parser.add_argument('no_agents', type=int, help='The number of agents to run in the grid environment', action='store')
    parser.add_argument('no_timesteps', type=int, help='The number of timesteps for the agent to explore', action='store')
    return parser.parse_args(args)



if __name__ == "__main__":
    #args = parse_args(sys.argv[1:])
    #agent_name = args.agent_name
    #print('args: ',args) 

    agent1_name = 'agent1'
    agent2_name = 'agent2'
    agent3_name = 'agent3'    
    #hard code grid for now
    grid = UE4Grid(20, 15, Vector3r(0,0), 120, 150)    

    
    agent1 = SimpleGridAgent(grid, Vector3r(0,0), get_move_from_belief_map_epsilon_greedy, -10, 0.3, agent1_name, other_active_agents = ['agent2'], comms_radius = 2)
    agent2 = SimpleGridAgent(grid, Vector3r(0,1), get_move_from_belief_map_epsilon_greedy, -10, 0.3, agent2_name, other_active_agents = ['agent1'], comms_radius = 2)
    #agent3 = SimpleGridAgent(grid, Vector3r(0,0), get_move_from_belief_map_epsilon_greedy, -10, 0.3, agent3_name)
        
    print("Running simulation with 2 agents")
    run_t_timesteps([agent1, agent2], 5)
    
    #grid, initial_pos, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, prior = {}
   # OccupancyGridAgent(grid, Vector3r(0,0), get_move_from_belief_map_epsilon_greedy, -12, 0.3, agent_name, other_active_agents).explore_t_timesteps(args.no_timesteps)
    
    
    #destroy_rav(client, agent_name)
























