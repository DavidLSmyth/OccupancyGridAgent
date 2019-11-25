# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:28:28 2019

@author: 13383861
"""

import pandas as pd
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

search_strategies = ['Sweep','Random', 'Saccadic', 'EpsilonGreedy']
no_agents = ['2', '3']
for search_strategy in search_strategies:
    for agent_no in no_agents:
        try:
            os.chdir("D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults/MultipleAgentsComms/{}/{}".format(agent_no,search_strategy))
            #csv_locs = ["MultipleAgentComms{}Process1.csv".format(i) for i in range(1, 2)]    
            csv_locs = list(filter(lambda x: x[-3:] == 'csv', list(os.walk('.'))[0][2]))
            print('csvs', csv_locs)
            #sys.exit()
            data_frames = [pd.read_csv(csv_loc, sep='\t') for csv_loc in csv_locs]
            print("no dataframes : {}".format(len(data_frames)))
            times = pd.concat([dframe['TTD'] for dframe in data_frames], axis = 1, keys = ['TTD{}'.format(_) for _ in range(len(data_frames))])
            concluded_locs = pd.concat([dframe['ConcludedLocation'] for dframe in data_frames], axis = 1, keys = ['ConcludedLoc{}'.format(_) for _ in range(len(data_frames))])
            for index, row in times.iterrows():
                #print(row[0])
                if row['TTD1'] not in [row['TTD0']+1, row['TTD0'], row['TTD0']-1]:
                    print("Error TTD doesn't match: {}".format(row))
                    
            for index, row in concluded_locs.iterrows():
                #print(row[0])
                if row['ConcludedLoc1'] != row['ConcludedLoc0']:
                    print("Error concluded locations don't match: {}".format(row))
            
            print("No concluded locs: {}".format(len(concluded_locs)))
            sys.exit()
            #all_data = pd.concat(data_frames)
            #list(all_data)
            #assert len(all_data) == 5000, str(len(all_data)) + ' ' + agent_no + ' ' +search_strategy
            #pos_decision_filter = all_data['ConcludedLocation'] != "Vector3r(-1.0, -1.0, 0.0)"
            #pos_data = all_data[pos_decision_filter]
            #neg_decision_filter = all_data['ConcludedLocation'] == "Vector3r(-1.0, -1.0, 0.0)"
            #neg_data = all_data[neg_decision_filter]
            
            #assert len(pos_data) + len(neg_data) == len(all_data)
            
            
        
        except Exception as e:
            print(f"Skipped generation of analysis for {search_strategy} {agent_no}")
            print(e)   
