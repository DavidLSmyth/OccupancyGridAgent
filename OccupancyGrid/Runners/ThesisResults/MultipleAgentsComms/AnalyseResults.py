# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:28:28 2019

@author: 13383861
"""

import pandas as pd
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix


search_strategies = ['Sweep','Random', 'EpsilonGreedy', 'Saccadic']
no_agents = ['2', '3']
for search_strategy in search_strategies:
    for agent_no in no_agents:
        try:
            os.chdir("D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults/MultipleAgentsComms/{}/{}".format(agent_no,search_strategy))
            #csv_locs = ["MultipleAgentComms{}Process1.csv".format(i) for i in range(1, 2)]    
            #csv_locs = ["MultipleAgentComms{}Process1.csv".format(i) for i in range(1, 2)]    
            csv_locs = list(filter(lambda x: x[-3:] == 'csv', list(os.walk('.'))[0][2]))
            print('csvs', csv_locs)
            #sys.exit()
            data_frames = [pd.read_csv(csv_loc, sep='\t') for csv_loc in csv_locs]
            print("no dataframes : {}".format(len(data_frames)))
            times = pd.concat([dframe['TTD'] for dframe in data_frames], axis = 1, keys = ['TTD{}'.format(_) for _ in range(len(data_frames))])
            concluded_locs = pd.concat([dframe['ConcludedLocation'] for dframe in data_frames], axis = 1, keys = ['ConcludedLoc{}'.format(_) for _ in range(len(data_frames))])
            
            non_matching_concluded_locs = 0
            non_matching_timesteps = 0
            for index, row in times.iterrows():
                #print(row[0])
                if len(data_frames) == 2:
                    if row['TTD1'] not in [row['TTD0']+1, row['TTD0'], row['TTD0']-1]:
                        non_matching_timesteps+=1
                elif len(data_frames) == 3:
                    if row['TTD1'] not in [row['TTD2']+1, row['TTD2'], row['TTD2']-1] or row['TTD1'] not in [row['TTD0']+1, row['TTD0'], row['TTD0']-1]:
                        non_matching_timesteps+=1
                    
            for index, row in concluded_locs.iterrows():
                #print(row[0])
                if len(data_frames) == 2:
                    if row['ConcludedLoc1'] != row['ConcludedLoc0']:
                        non_matching_concluded_locs+=1
                        
                elif len(data_frames) == 3:
                    if row['ConcludedLoc2'] != row['ConcludedLoc0'] or row['ConcludedLoc2'] != row['ConcludedLoc1']:
                        non_matching_concluded_locs+=1
                        
            #once metrics have been calculated, return the longest timestep conclusion
            all_data = pd.concat([times, concluded_locs, data_frames[0]["ActualLocation"]], axis = 1)
            if len(data_frames) == 2:
                all_data['TTD'] = all_data[["TTD0","TTD1"]].max(axis = 1)
            elif len(data_frames) == 3:
                all_data['TTD'] = all_data[["TTD0","TTD1", "TTD2"]].max(axis = 1)
            
            concluded_locs = []
            for index, row in all_data.iterrows():
                if len(data_frames) == 2:
                    if row["TTD0"] == row["TTD"]:
                        concluded_locs.append(row["ConcludedLoc0"])
                    elif row["TTD1"] == row["TTD"]:
                        concluded_locs.append(row["ConcludedLoc1"])
                    else:
                        raise Exception("Could not determine concluded location")
                        
                elif len(data_frames) == 3:
                    if row["TTD0"] == row["TTD"]:
                        concluded_locs.append(row["ConcludedLoc0"])
                    elif row["TTD1"] == row["TTD"]:
                        concluded_locs.append(row["ConcludedLoc1"])
                    elif row["TTD2"] == row["TTD"]:
                        concluded_locs.append(row["ConcludedLoc2"])
                    else:
                        raise Exception("Could not determine concluded location")
            all_data["ConcludedLocation"] = concluded_locs
            print(all_data.describe())
            print(all_data)
            #assert len(all_data) == 5000, str(len(all_data)) + ' ' + agent_no + ' ' +search_strategy
            pos_decision_filter = all_data['ConcludedLocation'] != "Vector3r(-1.0, -1.0, 0.0)"
            pos_data = all_data[pos_decision_filter]
            neg_decision_filter = all_data['ConcludedLocation'] == "Vector3r(-1.0, -1.0, 0.0)"
            neg_data = all_data[neg_decision_filter]
            
            assert len(pos_data) + len(neg_data) == len(all_data)
            
            expected_time_to_decision = np.mean(all_data['TTD'])
            expected_time_to_decision_pos = np.mean(pos_data['TTD'])
            expected_time_to_decision_neg = np.mean(neg_data['TTD'])
            
            var_ttd = np.var(all_data['TTD'])
            var_ttd_pos = np.var(pos_data['TTD'])
            var_ttd_neg = np.var(neg_data['TTD'])
            
            sd_ttd = np.std(all_data['TTD'])
            sd_ttd_pos = np.std(pos_data['TTD'])
            sd_ttd_neg = np.std(neg_data['TTD'])
            
            no_rejections = sum(all_data["ConcludedLocation"] == "Vector3r(-1.0, -1.0, 0.0)")
            #759
            prop_rejections = no_rejections / len(all_data)
            #This is the false negative rate
            #0.15
            
            no_incorrect_identifications = len(all_data) - sum( i == j for i,j in zip(list(all_data['ConcludedLocation'] != "Vector3r(-1.0, -1.0, 0.0)"), list(all_data["ConcludedLocation"] == all_data["ActualLocation"])))
            prop_incorrect_identifications = no_incorrect_identifications / len(all_data)
            #185 incorrect positive identifications
            #This is not the false positive rate!
            #185/5000
            no_rejections = sum(all_data["ConcludedLocation"] == "Vector3r(-1.0, -1.0, 0.0)")
            #759
            prop_rejections = no_rejections / len(all_data)
            #This is the false negative rate
            #0.15
            
            no_incorrect_identifications = len(all_data) - sum( i == j for i,j in zip(list(all_data['ConcludedLocation'] != "Vector3r(-1.0, -1.0, 0.0)"), list(all_data["ConcludedLocation"] == all_data["ActualLocation"])))
            prop_incorrect_identifications = no_incorrect_identifications / len(all_data)
            #185 incorrect positive identifications
            #This is not the false positive rate!
            #185/5000
            all_data['NoGuesses'] = all_data['ConcludedLocation'].apply(lambda x: len(re.findall('(Vector.*?\))', x)) - Counter(re.findall('(Vector.*?\))', x)).get('Vector3r(-1.0, -1.0, 0.0)', 0))
        
            all_data['NoCorrectGuesses'] = [len(set(a).intersection(set(b))) for a,b in zip(all_data['ConcludedLocation'].apply(lambda x: re.findall('(Vector.*?\))', x)), all_data['ActualLocation'].apply(lambda x: re.findall('(Vector.*?\))', x)))]
            conf_matrix = confusion_matrix(all_data['NoGuesses'], all_data['NoCorrectGuesses'],labels = [0,1,2,3])
            
            results_file = 'Results.json'
            with open(results_file,'w') as f:
                f.write('ETTD: {}\n'.format(expected_time_to_decision))
                f.write('ETTDPos: {}\n'.format(expected_time_to_decision_pos))
                f.write('ETTDNeg: {}\n'.format(expected_time_to_decision_neg))

                f.write('Var TTD: {}\n'.format(var_ttd))
                f.write('Var TTDPos: {}\n'.format(var_ttd_pos))
                f.write('Var TTDNeg: {}\n'.format(var_ttd_neg))
                
                f.write('SD TTD: {}\n'.format(sd_ttd))
                f.write('SD TTDPos: {}\n'.format(sd_ttd_pos))
                f.write('SD TTDNeg: {}\n'.format(sd_ttd_neg))
                
                f.write("NoRejections: {}\n".format(no_rejections))
                f.write("ProportionRejections: {}\n".format(prop_rejections))
                f.write("NumberIncorrectIdentifications: {}\n".format(no_incorrect_identifications))
                f.write("PropIncorrectIdentifications: {}\n".format(prop_incorrect_identifications))
                f.write("ProportionBothAgentsDontAgreeTimeStep: {}\n".format(non_matching_timesteps/len(all_data)))
                f.write("ProportionBothAgentsDontAgreeConcludedLocation: {}\n".format(non_matching_concluded_locs/len(all_data)))
                f.write("ConfusionMatrixNormalizedLatex(rows=NoGuesses,Col=NoCorrectGuesses): {}\n".format('\n'.join([' & '.join(list(filter(lambda x: x!='',map(lambda x: x.strip(),str((conf_matrix/conf_matrix.sum())[_])[1:-1].split(' '))))) for _ in range(len(conf_matrix))])))
            
            #%%
            plt.clf()
            #hist, bin_edges = np.histogram(all_data['TTD'], bins = [10*_ for _ in range(30)])
            bin_interval = 5
            max_bins = math.ceil(all_data['TTD'].max()/bin_interval)
            
            hist = np.histogram(all_data['TTD'], bins=[5*_ for _ in range(max_bins)])
            hist[0].max()
            plot_data = [all_data[all_data['NoGuesses'] == 3]['TTD'], all_data[all_data['NoGuesses'] == 2]['TTD'], all_data[all_data['NoGuesses'] == 1]['TTD'], all_data[all_data['NoGuesses'] == 0]['TTD']]
            label = ["Three guesses at target location", "Two guesses at target location", "One guess at target location", "Zero guesses at target location"]
            color = ['red', 'blue', 'green', 'yellow']
            if len(all_data[all_data['NoGuesses'] == 2]['TTD']) ==0:
                plot_data = plot_data[2:]
                label = label[2:]
                color = color[2:]
            elif len(all_data[all_data['NoGuesses'] == 3]['TTD']) ==0:
                plot_data = plot_data[1:]
                label = label[1:]
                color = color[1:]
            plt.figure(figsize = (10, 8))
            hist_plot = plt.hist(plot_data,bins=[5*_ for _ in range(max_bins)], color = color, stacked = True, label = label) 
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Number of Timesteps')
            plt.ylabel('Frequency')
            plt.title('Histogram of Time To Decision Using {} Search Strategy'.format(search_strategy))
            plt.legend()
            plt.vlines(expected_time_to_decision, ymin = 0, ymax = hist[0].max()*1.025, label = "Mean Time To Decision", linestyles = 'dashed')
            plt.ylim(0,hist[0].max()*1.025)
            
            plt.savefig("D:\\OccupancyGrid\\OccupancyGrid\\Runners\\ThesisResults\\Histograms1\\SingleAgentSingleSource{}{}Histogram.png".format(agent_no, search_strategy))
        except Exception as e:
            print(f"Skipped generation of analysis for {search_strategy} {agent_no}")
            raise e
            print(e)
