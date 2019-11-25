# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:30:32 2019

@author: 13383861
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:28:28 2019

@author: 13383861
"""

import pandas as pd
import os
import math
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

search_strategies = ['Random', 'Saccadic', 'EpsilonGreedy', 'Sweep']
no_sources = ['2', '3']

def get_location(location_number):
    def get_concl_location(text):
        if text == 'NeitherPresent':
            return 'NeitherPresent'
        else:
            if location_number == 1:
                return 'V' + text.split('V')[1][:-1]
            elif location_number == 2:
                return 'V' + text.split('V')[2]
            else:
                raise Exception("Cannot retrieve location {}".format(location_number))
    return get_concl_location

#%%
for search_strategy in search_strategies:
    for source_no in no_sources:
        try:
            os.chdir("D:/OccupancyGrid/OccupancyGrid/Runners/ThesisResults/MultipleTarget/{}/{}".format(source_no,search_strategy))
            csv_locs = ["MultipleTargetProcess{}.csv".format(i) for i in range(1, 9)]    
            data_frames = [pd.read_csv(csv_loc, sep='\t') for csv_loc in csv_locs]
            all_data = pd.concat(data_frames)

            all_data['ConcludedLocation'] = all_data['ConcludedLocation'].apply(lambda x: 'NeitherPresent' if x == "Vector3r(-1.0, -1.0, 0.0)" else x)
            all_data['ConcludedLocation1'] = all_data["ConcludedLocation"].apply(get_location(1))
            all_data['ConcludedLocation2'] = all_data["ConcludedLocation"].apply(get_location(2))
            
            len(re.findall('(Vector.*?\))',all_data['ConcludedLocation'].iloc[0]))
            all_data['NoGuesses'] = all_data['ConcludedLocation'].apply(lambda x: len(re.findall('(Vector.*?\))', x)) - Counter(re.findall('(Vector.*?\))', x)).get('Vector3r(-1.0, -1.0, 0.0)', 0))
    
            all_data['NoCorrectGuesses'] = [len(set(a).intersection(set(b))) for a,b in zip(all_data['ConcludedLocation'].apply(lambda x: re.findall('(Vector.*?\))', x)), all_data['ActualLocation'].apply(lambda x: re.findall('(Vector.*?\))', x)))]
            
            conf_matrix = confusion_matrix(all_data['NoGuesses'], all_data['NoCorrectGuesses'],labels = [0, 1,2,3])
            print(conf_matrix )
    
#            all_data.iloc[72]
#            all_data.iloc[88]
#            all_data.iloc[253]
#            
#            all_data['ConcludedLocation'].apply(lambda x: set(re.findall('(Vector.*?\))', x))).intersect - Counter(re.findall('(Vector.*?\))', x)).get('Vector3r(-1.0, -1.0, 0.0)', 0))
#    
#            all_data['NoGuesses'].describe()
#            all_data.iloc[109]
#            sum(all_data['NoGuesses'] == 2)
#            sum(all_data['NoGuesses'] == 1)
#            sum(all_data['NoGuesses'] == 0)
#            all_data[all_data['NoGuesses'] == 0]
            
    
            all_data['ActualLocation1'] = all_data["ActualLocation"].apply(get_location(1))
            all_data['ActualLocation2'] = all_data["ActualLocation"].apply(get_location(2))
            #for debugging. If something goes wrong, can identify which simulation parameters were used
            assert len(all_data) == 5000, str(len(all_data)) + ' ' + source_no + ' ' +search_strategy
            
#            #conclusion that two targets present
#            #neither present condition is actually redundant
#            two_pos_decision_filter = (all_data.ConcludedLocation1.apply(lambda x: x not in ["NeitherPresent", "Vector3r(-1.0, -1.0, 0.0)"])) & all_data.ConcludedLocation2.apply(lambda x: x not in ["NeitherPresent", "Vector3r(-1.0, -1.0, 0.0)"])
#            two_pos_data = all_data[two_pos_decision_filter]
#            
#            #conclusion that one target present
#            one_pos_decision_filter = ((all_data.ConcludedLocation1.apply(lambda x: x != "Vector3r(-1.0, -1.0, 0.0)") & all_data.ConcludedLocation2.apply(lambda x: x == "Vector3r(-1.0, -1.0, 0.0)")) | (all_data.ConcludedLocation2.apply(lambda x: x != "Vector3r(-1.0, -1.0, 0.0)") & all_data.ConcludedLocation1.apply(lambda x: x == "Vector3r(-1.0, -1.0, 0.0)")))
#            one_pos_data = all_data[one_pos_decision_filter]
#
#            #conclusion that no target present
#            zero_pos_decision_filter = (all_data.ConcludedLocation1.apply(lambda x: x == "NeitherPresent") & all_data.ConcludedLocation2.apply(lambda x: x == "NeitherPresent"))
#            zero_pos_data = all_data[zero_pos_decision_filter]
#            
#            assert len(two_pos_data) + len(one_pos_data) + len(zero_pos_data) == 5000, len(two_pos_data) + len(one_pos_data) + len(zero_pos_data) 
#            
#            #%%
#            two_pos_data = all_data[two_pos_decision_filter]
#            number_both_correct = 0
#            for index, row in two_pos_data.iterrows():
#                if (row.ConcludedLocation1 == row.ActualLocation1 or row.ConcludedLocation1 == row.ActualLocation2) and (row.ConcludedLocation2 == row.ActualLocation1 or row.ConcludedLocation2 == row.ActualLocation2):
#                    number_both_correct += 1
#                
#            #want the proportion of data points which contain two correct guesses
#            proportion_both_correct = number_both_correct / len(all_data)
#                
#            number_one_correct = 0
#            for index, row in two_pos_data.iterrows():
#                if (row.ConcludedLocation1 == row.ActualLocation1 or row.ConcludedLocation1 == row.ActualLocation2) ^ (row.ConcludedLocation2 == row.ActualLocation1 or row.ConcludedLocation2 == row.ActualLocation2):
#                    number_one_correct += 1
#            
#            proportion_one_correct_one_incorrect = number_one_correct / len(all_data)
#            
#            number_zero_correct = 0
#            for index, row in two_pos_data.iterrows():
#                if (row.ConcludedLocation1 != row.ActualLocation1 and row.ConcludedLocation1 != row.ActualLocation2) and (row.ConcludedLocation2 != row.ActualLocation1 and row.ConcludedLocation2 != row.ActualLocation2):
#                    number_zero_correct += 1
#                    
#            proportion_both_incorrect = number_zero_correct / len(all_data)
#            
#            #%%
#            one_pos_data = all_data[one_pos_decision_filter]
#            print(len(one_pos_data))
#            number_one_correct_one_pos = 0
#            for index, row in one_pos_data.iterrows():
#                if (row.ConcludedLocation1 == row.ActualLocation1 or row.ConcludedLocation1 == row.ActualLocation2) ^ (row.ConcludedLocation2 == row.ActualLocation1 or row.ConcludedLocation2 == row.ActualLocation2):
#                    number_one_correct_one_pos += 1
#                
#            proportion_single_guess_correct = number_one_correct_one_pos / len(all_data)
#
#                
#            number_zero_correct_one_pos= 0
#            for index, row in one_pos_data.iterrows():
#                if (row.ConcludedLocation1 != row.ActualLocation1 and row.ConcludedLocation1 != row.ActualLocation2) and (row.ConcludedLocation2 != row.ActualLocation1 and row.ConcludedLocation2 != row.ActualLocation2):
#                    number_zero_correct_one_pos += 1
#                    
#            proportion_single_guess_incorrect = number_zero_correct_one_pos / len(all_data)
#
#            proportion_both_not_present = len(zero_pos_data) / len(all_data)
            
            #assert len(zero_pos_data)+number_both_correct+number_one_correct+number_zero_correct+number_one_correct_one_pos+number_zero_correct_one_pos==5000
            #%%
            
            #pos_data = all_data[pos_decision_filter]
            #neg_decision_filter = all_data['ConcludedLocation'] == "Vector3r(-1.0, -1.0, 0.0)"
            #neg_data = all_data[neg_decision_filter]
            
            #assert len(pos_data) + len(neg_data) == len(all_data)
            
            expected_time_to_decision = np.mean(all_data['TTD'])
            expected_time_to_decision_two_pos = np.mean(all_data[all_data['NoGuesses'] == 2]['TTD'])
            expected_time_to_decision_one_pos = np.mean(all_data[all_data['NoGuesses'] == 1]['TTD'])
            expected_time_to_decision_zero_pos = np.mean(all_data[all_data['NoGuesses'] == 0]['TTD'])
            
            sd_ttd = np.std(all_data['TTD'])
            sd_ttd_two_pos = np.std(all_data[all_data['NoGuesses'] == 2]['TTD'])
            sd_ttd_one_pos = np.std(all_data[all_data['NoGuesses'] == 1]['TTD'])
            sd_ttd_zero_pos = np.std(all_data[all_data['NoGuesses'] == 0]['TTD'])
            
            #no_rejections = sum(all_data["ConcludedLocation"] == "Vector3r(-1.0, -1.0, 0.0)")
            #759
            #prop_rejections = no_rejections / len(all_data)
            #This is the false negative rate
            #0.15
            
            #no_incorrect_identifications = len(all_data) - sum( i == j for i,j in zip(list(all_data['ConcludedLocation'] != "Vector3r(-1.0, -1.0, 0.0)"), list(all_data["ConcludedLocation"] == all_data["ActualLocation"])))
            #prop_incorrect_identifications = no_incorrect_identifications / len(all_data)
            #185 incorrect positive identifications
            #This is not the false positive rate!
            #185/5000
            
            results_file = 'Results.json'
            with open(results_file,'w') as f:
                f.write('ETTDPos: {}\n'.format(expected_time_to_decision))
                if source_no == '3':
                    f.write('ETTDThreePos: {}\n'.format(np.mean(all_data[all_data['NoGuesses'] == 3]['TTD'])))
                f.write('ETTDTwoPos: {}\n'.format(expected_time_to_decision_two_pos))
                f.write('ETTDOnePos: {}\n'.format(expected_time_to_decision_one_pos))
                f.write('ETTDZeroPos: {}\n'.format(expected_time_to_decision_zero_pos))
                
                f.write('SDTTD: {}\n'.format(sd_ttd))
                if source_no == '3':
                    f.write('SDThreePos: {}\n'.format(np.std(all_data[all_data['NoGuesses'] == 3]['TTD'])))
                f.write('SDTwoPos: {}\n'.format(sd_ttd_two_pos))
                f.write('SDOnePos: {}\n'.format(sd_ttd_one_pos))
                f.write('SDZeroPos: {}\n'.format(sd_ttd_zero_pos))
                
                #f.write("NoRejections: {}\n".format(no_rejections))
                #f.write("ProportionRejections: {}\n".format(prop_rejections))
                #f.write("NumberIncorrectIdentifications: {}\n".format(no_incorrect_identifications))
                #f.write("PropIncorrectIdentifications: {}\n".format(prop_incorrect_identifications))
                
                #f.write("PropBothGuessesCorrect: {}\n".format(proportion_both_correct))
                #f.write("PropOneGuessCorrectOneIncorrect: {}\n".format(proportion_one_correct_one_incorrect))
                #f.write("PropBothGuessesIncorrect: {}\n".format(proportion_both_incorrect))
                #f.write("PropOneGuessCorrectSecondNotPresent: {}\n".format(proportion_single_guess_correct))
                #f.write("PropOneGuessInCorrectSecondNotPresent: {}\n".format(proportion_single_guess_incorrect))
                #f.write("PropBothNotPresent: {}\n".format(proportion_both_not_present))
                
                f.write("ConfusionMatrix(rows=NoGuesses,Col=NoCorrectGuesses): {}\n".format(str(conf_matrix)))
            #%%
            plt.clf()
            #hist, bin_edges = np.histogram(all_data['TTD'], bins = [10*_ for _ in range(30)])
            bin_interval = 5
            max_bins = math.ceil(all_data['TTD'].max()/bin_interval)
            
            hist = np.histogram(all_data['TTD'], bins=[5*_ for _ in range(max_bins)])
            hist[0].max()
            histogram_plot = all_data['TTD'].hist(bins=[5*_ for _ in range(max_bins)], color = 'blue')
            #two_pos_data['TTD'].hist(bins=[5*_ for _ in range(max_bins)], color = 'blue', label = 'Two Guesses at Target Location')
            #one_pos_data['TTD'].hist(bins=[5*_ for _ in range(max_bins)], color = 'red', label = 'One Guess at Target Location')
            #zero_pos_data['TTD'].hist(bins=[5*_ for _ in range(max_bins)], color = 'yellow', label = 'Zero Guesses at Target Location')
            plt.vlines(expected_time_to_decision, ymin = 0, ymax = hist[0].max()*1.025, label = "Mean Time To Decision", linestyles = 'dashed')
            
            #n, bins, patches = plt.hist(x=hist, bins=[10*_ for _ in range(30)], color='#0504aa',alpha=0.7, rwidth=0.85)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of Time To Decision Using {} Search Strategy'.format(search_strategy))
            plt.legend()
            plt.ylim(0,hist[0].max()*1.025)
            
            plt.savefig("SingleAgentSingleSource{}{}Histogram.png".format(source_no, search_strategy))
        except Exception as e:
            print(f"Skipped generation of analysis for {search_strategy} {source_no}")
            print(e)
            