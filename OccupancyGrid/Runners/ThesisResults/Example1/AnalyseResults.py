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
import re

base_dir = 'D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults'
os.chdir(base_dir)
subdirs = os.walk(base_dir)[0][1]
search_strategies = ['Random', 'Saccadic', 'EpsilonGreedy', 'Sweep']

for subdir in subdirs:
    for variable_param in os.walk(subdir)[1]:
	    for search_strategy in search_strategies:
		    print("changing dir to {}".format(base_dir + subdir + variable_param + search_strategy)

sys.exit()
varying_param = 
priors = ['Uniform', 'Gaussian']
for search_strategy in search_strategies:
    for prior in priors:
        os.chdir("D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults/Example1/{}/{}".format(prior,search_strategy))
        csv_locs = ["Experiment1Process{}.csv".format(i) for i in range(1, 9)]    
        data_frames = [pd.read_csv(csv_loc, sep='\t') for csv_loc in csv_locs]
        all_data = pd.concat(data_frames)
        list(all_data)
        assert len(all_data) == 5000, str(len(all_data)) + ' ' + prior + ' ' +search_strategy
        
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
            
        conf_matrix = confusion_matrix(all_data['NoGuesses'], all_data['NoCorrectGuesses'],labels = [0, 1,2,3])
        
        expected_time_to_decision = np.mean(all_data['TTD'])
        expected_time_to_decision_two_pos = np.mean(all_data[all_data['NoGuesses'] == 2]['TTD'])
        expected_time_to_decision_one_pos = np.mean(all_data[all_data['NoGuesses'] == 1]['TTD'])
        expected_time_to_decision_zero_pos = np.mean(all_data[all_data['NoGuesses'] == 0]['TTD'])
        
        sd_ttd = np.std(all_data['TTD'])
        sd_ttd_two_pos = np.std(all_data[all_data['NoGuesses'] == 2]['TTD'])
        sd_ttd_one_pos = np.std(all_data[all_data['NoGuesses'] == 1]['TTD'])
        sd_ttd_zero_pos = np.std(all_data[all_data['NoGuesses'] == 0]['TTD'])
        var_ttd = np.var(all_data['TTD'])
        sd_ttd = np.std(all_data['TTD'])
        
    
        
        
        results_file = 'Results.json'
        with open(results_file,'w') as f:
            f.write('ETTDPos: {}\n'.format(expected_time_to_decision))
            f.write('ETTDThreePos: {}\n'.format(np.mean(all_data[all_data['NoGuesses'] == 3]['TTD'])))
            f.write('ETTDTwoPos: {}\n'.format(expected_time_to_decision_two_pos))
            f.write('ETTDOnePos: {}\n'.format(expected_time_to_decision_one_pos))
            f.write('ETTDZeroPos: {}\n'.format(expected_time_to_decision_zero_pos))
            
            f.write('SDTTD: {}\n'.format(sd_ttd))
            f.write('SDThreePos: {}\n'.format(np.std(all_data[all_data['NoGuesses'] == 3]['TTD'])))
            f.write('SDTwoPos: {}\n'.format(sd_ttd_two_pos))
            f.write('SDOnePos: {}\n'.format(sd_ttd_one_pos))
            f.write('SDZeroPos: {}\n'.format(sd_ttd_zero_pos))
            f.write("NoRejections: {}\n".format(no_rejections))
            f.write("ProportionRejections: {}\n".format(prop_rejections))
            f.write("NumberIncorrectIdentifications: {}\n".format(no_incorrect_identifications))
            f.write("PropIncorrectIdentifications: {}\n".format(prop_incorrect_identifications))
            f.write("ConfusionMatrix(rows=NoGuesses,Col=NoCorrectGuesses): {}\n".format(str(conf_matrix)))
            
        
        #%%
        plt.clf()
        #hist, bin_edges = np.histogram(all_data['TTD'], bins = [10*_ for _ in range(30)])
        bin_interval = 5
        max_bins = math.ceil(all_data['TTD'].max()/bin_interval)
        #plot data 
        histogram_plot = all_data['TTD'].hist(bins=[5*_ for _ in range(max_bins)], color = 'blue')
        
        #n, bins, patches = plt.hist(x=hist, bins=[10*_ for _ in range(30)], color='#0504aa',alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Time To Decision Using {} Search Strategy'.format(search_strategy))
        #plt.text(23, 45, r'$\mu=15, b=3$')
        # Set a clean upper y-axis limit.
        #plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        
        plt.savefig("SingleAgentSingleSource{}{}Histogram.png".format(prior, search_strategy))
