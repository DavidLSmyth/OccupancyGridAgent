# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 19:30:42 2019

@author: 13383861
"""


#prior should be a map from typing.Dict[Vector3r, float]
import typing
import math

from scipy.stats import multivariate_normal
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from Utils.UE4Grid import UE4Grid
from Utils.Vector3r import Vector3r
#for testing purposes


#%%
class Prior:
    def __init__(self, grid, distribution_generator: "A function that takes in a grid location and returns the corresponding prior pdf value"):
        self.prior_dict = {grid_loc: distribution_generator(grid_loc) for grid_loc in grid.get_grid_points()}
        
    def get_prior_dict(self):
        return self.prior_dict
        
    def show_prior(self):
        '''Shows prior on 3d graph'''
        pass
    
def _generate_gaussian_prior(grid: UE4Grid, means: "list of 2 means", covariance_matrix: "2x2 covariance matrix", initial_belief_sum = 0.5) -> "Tuple(np.array normed prior probs, prior_dict":
    '''A method that returns the normed z vector as well as the prior dict. Given a 2d vector of means, 2X2 covariance matrix, returns a 2D gaussian prior'''
    #maybe should check all dimensions of data here
    prior = {}
    x, y = np.mgrid[0:grid.get_no_points_x() * grid.get_lng_spacing():grid.get_lng_spacing(), 0:grid.get_no_points_y() * grid.get_lat_spacing():grid.get_lat_spacing()]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv = multivariate_normal(means, covariance_matrix)
    z = rv.pdf(pos)
    normed_z = z * (initial_belief_sum/z.sum())
    
    #normed_z = normed_z.astype(float)
    #return likelihoods in dict with grid points
    for x_value in [_[0] for _ in x]:
        for y_value in y[0]:
            prior[Vector3r(x_value, y_value)] = float(normed_z[x_value][y_value])
    
    #place prior into ordered numpy array
    list_of_grid_points = []
    for grid_loc in grid.get_grid_points():
        list_of_grid_points.append(prior[grid_loc])
        
    list_of_grid_points = np.array(list_of_grid_points)
    numpy_prior = np.append(list_of_grid_points, 1 - list_of_grid_points.sum())
    return numpy_prior
    
def generate_gaussian_prior(grid: UE4Grid, means: "list of 2 means", covariance_matrix: "2x2 covariance matrix", initial_belief_sum) -> "Tuple(np.array normed prior probs, prior_dict":
    '''Given a 2d vector of means, 2X2 covariance matrix, returns a 2D gaussian prior'''
    return _generate_gaussian_prior(grid, means, covariance_matrix, initial_belief_sum)


def generate_uniform_prior(grid: UE4Grid, initial_belief_sum, fixed_value = None):
    '''Generates a uniform prior which equates to delta/|grid|. The sum of each grid location prior is equal to initial_belief_sum. If fixed_value provided, 
    each location in the prior will have fixed_value as prior'''
    if fixed_value and 0<=fixed_value<=1:
        prior_val = fixed_value
    else:
        prior_val = initial_belief_sum/len(grid.get_grid_points())
    return np.array([prior_val for grid_loc in grid.get_grid_points()] + [1 - prior_val * grid.get_no_grid_points()])
    
    
def plot_prior(grid: UE4Grid, prior):
    fig = plt.figure()
    x, y = np.mgrid[0:grid.get_no_points_x() * grid.get_lng_spacing():grid.get_lng_spacing(), 0:grid.get_no_points_y() * grid.get_lat_spacing():grid.get_lat_spacing()]
    ax = fig.gca(projection='3d')
    prior = prior[:-1]
    z_lim = prior.max() * 1.5
    ax.set_zlim3d(0, z_lim)
    if prior.shape != (len(x),len(y)):
        prior = prior.reshape(len(x),len(y))
    ax.plot_wireframe(x, y, prior)
    ax.set_xlabel("Physical x-axis of grid")
    ax.set_ylabel("Physical y-axis of grid")
    #ax.set_zlabel("Initial belief target is present at grid location")
    plt.title("Initial belief distribution over the region of interest")
    return fig
    
def generate_and_save_prior_fig(grid: UE4Grid, prior, file_path):
    fig = plot_prior(grid, prior)
    fig.savefig(file_path)
    
#def save_gaussian_prior(grid: UE4Grid, means: "list of 2 means", covariance_matrix: "2x2 covariance matrix", file_path, initial_belief_sum = 0.5):
#    prior = _generate_gaussian_prior(grid, means, covariance_matrix, initial_belief_sum)[0]
#    fig = plt.figure()
#    x, y = np.mgrid[0:grid.get_no_points_x() * grid.get_lng_spacing():grid.get_lng_spacing(), 0:grid.get_no_points_y() * grid.get_lat_spacing():grid.get_lat_spacing()]
#    ax = fig.gca(projection='3d')
#    z_lim = prior.max() * 1.02
#    ax.set_zlim3d(0, z_lim)
#    ax.plot_wireframe(x, y, prior)
#    plt.savefig(file_path)
    
def is_valid_initial_dist(no_grid_points, initial_dist):
    '''Verifies that the specified initial distribution over a grid is valid. It's assumed that the initial distribution
    represents the probability of evidence being located at a particular grid location.'''
    return math.isclose(initial_dist.sum(), 1, rel_tol = 0.0000001) and len(initial_dist) == no_grid_points + 1
    
#%%
if __name__ == '__main__':
#%%

    grid = UE4Grid(1, 1, Vector3r(0,0), 9, 9)
    grid.get_grid_points()
    means = [3,4]
    covariance_matrix = [[3.0, 0], [0, 3]]
    initial_belief_sum = 0.5
    gaussian_prior = _generate_gaussian_prior(grid, means, covariance_matrix, initial_belief_sum)
    plot_prior(grid, gaussian_prior)
    assert is_valid_initial_dist(grid.get_no_grid_points(), gaussian_prior)
    
    prior = generate_gaussian_prior(grid, means, covariance_matrix, initial_belief_sum = 0.5)
    assert 0.49 < prior[:-1].sum() < 0.51
    
    
#%%    
    uniform_prior = generate_uniform_prior(grid, initial_belief_sum = 0.8)
    assert is_valid_initial_dist(grid.get_no_grid_points(), uniform_prior)
    plot_prior(grid, uniform_prior)
    plot_prior(grid, gaussian_prior)
    
    from Utils.BeliefMap import create_single_source_binary_belief_map
    #(grid, prior: np.array, fpr = 0.2, fnr = 0.1)
    single_source_bel_map = create_single_source_binary_belief_map(grid, prior, fpr = 0.2, fnr = 0.1)
    


