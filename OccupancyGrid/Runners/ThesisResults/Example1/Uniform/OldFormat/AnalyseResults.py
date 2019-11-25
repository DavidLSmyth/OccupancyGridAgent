# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:28:28 2019

@author: 13383861
"""

import pandas as pd
import os
import numpy as np
import re
import matplotlib.pyplot as plt

os.chdir("D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults/Example1/UniformPrior")

csv_locs = ["Experiment1Process{}.csv".format(i) for i in range(1, 9)]
data_frames = [pd.read_csv(csv_loc, sep='\t') for csv_loc in csv_locs]
all_data = pd.concat(data_frames)
list(all_data)
assert len(all_data) == 5000
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
    


#%%

all_data['TTD'].max()
all_data['TTD'].min()
all_data['TTD'].mean()
len(all_data['TTD'].unique())


#hist, bin_edges = np.histogram(all_data['TTD'], bins = [10*_ for _ in range(30)])

histogram_plot = all_data['TTD'].hist(bins=[5*_ for _ in range(60)], color = 'blue')

#n, bins, patches = plt.hist(x=hist, bins=[10*_ for _ in range(30)], color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Time To Decision.')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.savefig("SingleAgentSingleSourceUniformHistogram.png")
