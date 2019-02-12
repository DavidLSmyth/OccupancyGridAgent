# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:00:30 2018

@author: 13383861
"""

from collections import namedtuple
import sys
#update path so other modules can be imported
sys.path.append('..')
import typing
import csv
#from AirSimInterface.types import Vector3r
from Utils.Vector3r import Vector3r
#everything that could be important for measuring agent performance/progress
_AgentAnalysisState = typing.NamedTuple('AgentAnalysisState', [('position_intended',Vector3r),
                                                     ('position_measured',Vector3r),
                                                     ('timestep', int),
                                                     ('timestamp', float),
                                                     ('rav_name',str),
                                                     #maybe add distance travelled for current timestep
                                                     ('total_dist_travelled', float),
                                                     ('remaining_batt_cap', float),
                                                     ('prop_battery_cap_used', float),
                                                     ('sensor_reading', float),
                                                     #is it necessary to record the grid along with the likelihoods in case want the grid to 
                                                     #dynamically change? For now assume grid is fixed and in 1-1 correspondance with likelihoods
                                                     #'occ_grid_likelihoods',
                                                     #which other agents did the agent coordinate with on this timestep
                                                     ('coordinated_with_other_names', list)])

class AgentAnalysisState(_AgentAnalysisState):
    '''A wrapper class of _AgentAnalysisState to enforce correct data types'''
    def __new__(cls, position_intended,position_measured,timestep,timestamp,rav_name,total_dist_travelled,remaining_batt_cap,prop_battery_cap_used,sensor_reading,coordinated_with_other_names) -> '_AgentAnalysisState':
        args = (position_intended,position_measured,timestep,timestamp,rav_name,total_dist_travelled,remaining_batt_cap,prop_battery_cap_used,sensor_reading,coordinated_with_other_names)
        return super().__new__(cls, *[d_type(value) if type(value) is not d_type else value for value, d_type in zip(args, _AgentAnalysisState.__annotations__.values())])

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.__dict__ == other.__dict__
    def __hash__(self):
        return hash((self.__dict__.values()))
#metadata related to the agent - details about grid its operating in, prior that was worked with, to be updated...
AgentAnalysisMetadata= namedtuple("MissionAnalysisData", ["agents_used", "grid_origin_lng", "grid_origin_lat",'grid_lat_spacing', 
                                                         'grid_lng_spacing','lng_lim', 'lat_lim',
                                                         'no_lat_points', 'no_lng_points', 'prior'])

def get_agent_state_for_analysis(agent_analysis_state: AgentAnalysisState):
    '''Returns elements of agent state that are important for analysis that can be written to csv. Position, battery cap., total_dist_travelled, battery_consumed, occ_grid'''
    #csv_headers = ['timestep', 'timestamp', 'rav_name', 'position_intended', 'position_measured', 'total_dist_travelled', 'remaining_batt_cap', 'prop_battery_cap_used', 'sensor_reading', 'occ_grid_locs', 'occ_grid_likelihoods', 'coordinated_with_other_bool', 'coordinated_with_other_names']
    #return str(agent_analysis_state._fields).replace(')','').replace('(','').replace("'", '')
    return _get_agent_state_for_analysis(**agent_analysis_state._asdict())

def _get_agent_state_for_analysis(position_intended,position_measured,timestep,timestamp,rav_name,total_dist_travelled,remaining_batt_cap,prop_battery_cap_used,sensor_reading,coordinated_with_other_names):
    return f"{position_intended.x_val},{position_intended.y_val},{position_intended.z_val},{position_measured.x_val},{position_measured.y_val},{position_measured.z_val},{timestep},{timestamp},{rav_name},{total_dist_travelled},{remaining_batt_cap},{prop_battery_cap_used},{sensor_reading},{'|'.join(coordinated_with_other_names)}"
 
def get_agent_analysis_file_header():
    return "{position_intended.x_val},{position_intended.y_val},{position_intended.z_val},{position_measured.x_val},{position_measured.y_val},{position_measured.z_val},{timestep},{timestamp},{rav_name},{total_dist_travelled},{remaining_batt_cap},{prop_battery_cap_used},{sensor_reading},{coordinated_with_other_names}".replace("{",'').replace("}","")
        
def _init_state_for_analysis_file(file_path):
    with open(file_path, 'w+') as f:
        f.write(get_agent_analysis_file_header())

def _update_state_for_analysis_file(file_path, agent_analysis_state: AgentAnalysisState):
    with open(file_path, 'a') as f:
        f.write('\n'+get_agent_state_for_analysis(agent_analysis_state))
        
def _init_agent_metadata_file(file_path):
    with open(file_path, 'w+') as f:
        f.write(','.join(AgentAnalysisMetadata._fields) + '\n')

def _write_to_agent_metadata_file(file_path, meta_data: AgentAnalysisMetadata):
    with open(file_path, 'a') as f:
        f.write(''.join()  + '\n')
        
def read_agent_state_for_analysis_file(file_path: str)->typing.List[AgentAnalysisState]:
    #maybe have a type mapping for this
    try:
        with open(file_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            return [AgentAnalysisState(Vector3r(row[0], row[1], row[2]), Vector3r(row[3], row[4], row[5]), *row[6:-1], list(row[-1].split('|'))) for row in reader]
    except Exception as e:
        raise e

if __name__ == "__main__":
    testAgentAnalysisState1 = AgentAnalysisState(Vector3r(10,20,30), Vector3r(40,50,60), 2, 1.1252, 'rav_name',3.1,2.8,2.8,2.6,['drone1', 'drone2'])
    testAgentAnalysisState2 = AgentAnalysisState(Vector3r(70,20,12), Vector3r(12.5,50,60.2), 2, 1.1252, 'rav_name',3.1,2.8,2.8,2.1, ['drone1'])
    testAgentAnalysisState3 = AgentAnalysisState(Vector3r(0.0,15.0,0.0), Vector3r(25.0036563873291, -2.3688066005706787, -11.998088836669922), 0, 1542291931.2268043, "Drone2", 15.0, 0.9784099446151149, 0.02159005538488512, 0.0111930622, ['Drone1'])
    
    testAgentAnalysisState1._asdict()
    
    assert get_agent_state_for_analysis(testAgentAnalysisState1) == "10.0,20.0,30.0,40.0,50.0,60.0,2,1.1252,rav_name,3.1,2.8,2.8,2.6,drone1|drone2"
    assert get_agent_analysis_file_header() == 'position_intended.x_val,position_intended.y_val,position_intended.z_val,position_measured.x_val,position_measured.y_val,position_measured.z_val,timestep,timestamp,rav_name,total_dist_travelled,remaining_batt_cap,prop_battery_cap_used,sensor_reading,coordinated_with_other_names'
    
    mock_data_fp = "D:/ReinforcementLearning/DetectSourceAgent/Data/MockData/testAgentAnalysisState.csv"
    _init_state_for_analysis_file(mock_data_fp)
    _update_state_for_analysis_file(mock_data_fp, testAgentAnalysisState1)
    _update_state_for_analysis_file(mock_data_fp, testAgentAnalysisState2)
    
    assert testAgentAnalysisState1 in read_agent_state_for_analysis_file(mock_data_fp)
    
    
    
    

    