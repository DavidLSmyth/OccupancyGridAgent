# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:28:28 2019

@author: 13383861
"""

import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt

search_strategies = ['Random', 'Saccadic', 'EpsilonGreedy', 'Sweep']
sensor_params = ['4-4', '05-02']
for search_strategy in search_strategies:
    for sensor_param in sensor_params:
        try:
            os.chdir("D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults/MiscalibratedSensor/{}/{}".format(sensor_param,search_strategy))
            csv_locs = ["MiscalibratedSensorProcess{}.csv".format(i) for i in range(1, 9)]    
            data_frames = [pd.read_csv(csv_loc, sep='\t') for csv_loc in csv_locs]
            all_data = pd.concat(data_frames)
            list(all_data)
            assert len(all_data) == 5000, str(len(all_data)) + ' ' + sensor_param + ' ' +search_strategy
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
                
            
            #%%
            plt.clf()
            #hist, bin_edges = np.histogram(all_data['TTD'], bins = [10*_ for _ in range(30)])
            bin_interval = 5
            max_bins = math.ceil(all_data['TTD'].max()/bin_interval)
            
            hist = np.histogram(all_data['TTD'], bins=[5*_ for _ in range(max_bins)])
            hist[0].max()
            histogram_plot = all_data['TTD'].hist(bins=[5*_ for _ in range(max_bins)], color = 'blue')
            pos_data['TTD'].hist(bins=[5*_ for _ in range(max_bins)], color = 'blue', label = 'Positive Decision')
            neg_data['TTD'].hist(bins=[5*_ for _ in range(max_bins)], color = 'red', label = 'Negative Decision')
            plt.vlines(expected_time_to_decision, ymin = 0, ymax = hist[0].max()*1.025, label = "Mean Time To Decision", linestyles = 'dashed')
            
            #n, bins, patches = plt.hist(x=hist, bins=[10*_ for _ in range(30)], color='#0504aa',alpha=0.7, rwidth=0.85)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of Time To Decision Using {} Search Strategy'.format(search_strategy))
            plt.legend()
            plt.ylim(0,hist[0].max()*1.025)
            
            plt.savefig("SingleAgentSingleSource{}{}Histogram.png".format(sensor_param[2:], search_strategy))
        except Exception as e:
            print(f"Skipped generation of analysis for {search_strategy} {sensor_param}")
