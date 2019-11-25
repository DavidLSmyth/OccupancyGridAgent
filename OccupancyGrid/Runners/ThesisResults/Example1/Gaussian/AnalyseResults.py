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
for search_strategy in search_strategies:
    
    os.chdir("D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults/Example1/GuassianPrior/{}".format(search_strategy))
    
    csv_locs = ["Experiment1Process{}.csv".format(i) for i in range(1, 9)]
    data_frames = [pd.read_csv(csv_loc, sep='\t') for csv_loc in csv_locs]
    all_data = pd.concat(data_frames)
    list(all_data)
    expected_time_to_decision = np.mean(all_data['TTD'])
    no_rejections = sum(all_data["ConcludedLocation"] == "Vector3r(-1.0, -1.0, 0.0)")
    #759
    prop_rejections = no_rejections / len(all_data)
    #This is the false negative rate
    #0.15
    
    no_incorrect_identifications = len(all_data) - sum( i == j for i,j in zip(list(all_data['ConcludedLocation'] != "Vector3r(-1.0, -1.0, 0.0)"), list(all_data["ConcludedLocation"] == all_data["ActualLocation"])))
    #185 incorrect positive identifications
    #This is not the false positive rate!
    #185/5000
    
    results_file = 'Results.json'
    with open(results_file,'w') as f:
        f.write('ETTD: {}\n'.format(expected_time_to_decision))
        f.write("NoRejections: {}\n".format(no_rejections))
        f.write("ProportionRejections: {}\n".format(prop_rejections))
        f.write("NumberIncorrectIdentifications: {}\n".format(no_incorrect_identifications))
        
    
    #%%
    
    #hist, bin_edges = np.histogram(all_data['TTD'], bins = [10*_ for _ in range(30)])
    bin_interval = 5
    max_bins = math.ceil(all_data['TTD'].max()/bin_interval)
    histogram_plot = all_data['TTD'].hist(bins=[5*_ for _ in range(max_bins)], color = 'blue')
    
    #n, bins, patches = plt.hist(x=hist, bins=[10*_ for _ in range(30)], color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time To Decision Using {} Search Strategy'.format(search_strategy))
    #plt.text(23, 45, r'$\mu=15, b=3$')
    # Set a clean upper y-axis limit.
    #plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    
    plt.savefig("SingleAgentSingleSourceGaussianHistogram.png")