# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:18:49 2019

@author: 13383861

Generates the code for julia transition model, observation model and initial distribution


"""

#%%
import os
def add_to_clipboard(text):
    command = 'echo | set /p nul=' + text.strip() + '| clip'
    os.system(command)
    
#%%
######################Generate transition matrix string######################
def generate_t_matrix_row(no_states, end_state):
    return '0 '*end_state + '1 ' + '0 '*(no_states-end_state-1) + ';'

def generate_t_matrix_values(no_states, end_state):
    '''
    Returns a string of the form 
    1 0 0 ... 0;
    .
    .
    1 0 0 ... 0;
    '''
    return '\n\t'.join([generate_t_matrix_row(no_states, end_state) for _ in range(no_states - 1)])

def generate_deteministic_t(no_states):
    '''
    Returns a string that can be copy and pasted into julia: 
    T[1, : , :] = [1 0 ... 0;
                   1 0 ... 0;
                   .
                   .
                   .
                   ]
    etc.
    '''
    return_string = ''
    base_string = 'T[{}, :, :] = [{}]\n\n'
    for end_state in range(no_states):
        return_string += base_string.format(end_state+1, generate_t_matrix_values(no_states, end_state))
    return return_string

######################Generate transition matrix string######################

######################Generate observation matrix string######################
#%%
def generate_o_matrix_row(no_states, observation, current_state):
    
    if observation == 1:
        return 'fpr '*current_state + 'tpr ' + 'fpr '*(no_states - current_state - 1) + ';'
    else:
        return 'tnr '*current_state + 'fnr ' + 'tnr '*(no_states - current_state - 1) + ';'
    
def generate_o_matrix(no_states, observation):
    '''
    Returns a string of the form 
    1 0 0 ... 0;
    .
    .
    1 0 0 ... 0;
    '''
    return '\n\t'.join([generate_o_matrix_row(no_states, observation, end_state) for end_state in range(no_states - 1)])

def generate_o(no_states):
    '''
    Returns a string that can be copy and pasted into julia: 
    O[1, : , :] = [fnr tnr ... tnr;
                   tnr fnr ... tnr;
                   .
                   .
                   .
                   ]
    etc.
    '''
    return_string = ''
    base_string = 'O[{}, :, :] = [{}]\n\n'
    for observation in range(2):
        return_string += base_string.format(observation+1, generate_o_matrix(no_states, observation))
    return return_string

######################Generate observation matrix string######################

######################Generate reward matrix string######################
    #%%
def generate_r_matrix_row(no_states, current_state):
    return '-1 '*current_state + ('100 ' if current_state != no_states else '') + '-1 '*(no_states - current_state - 1) + ';'
    
def generate_r_matrix(no_states):
    '''
    Returns a string of the form 
    100 -1 -1 ... -1;
    -1 100 -1 ... -1;
    .
    .
    -1 -1 -1 ... -1;
    '''
    return 'R = [' + '\n\t'.join([generate_r_matrix_row(no_states-1, end_state) for end_state in range(no_states)]) + ']'



######################Generate reward matrix string######################
#%%

#5x6 example
no_states = 101
with open('D:/OccupancyGrid/JuliaPOMDP/julia_t_o_matrices.txt', 'w') as f_out:
    f_out.write(generate_deteministic_t(no_states) + generate_o(no_states) + generate_r_matrix(no_states))











