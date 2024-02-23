"""
Compute the cost bound in Proposition 2 of the EPG paper
"""

import pdb, pickle, sys, itertools
#import yaml
from yaml.loader import SafeLoader
import argparse
#from loguru import logger
from copy import deepcopy

import numpy as np
from scipy import integrate
from scipy.special import softmax

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

## Simulation parameters
N_strategy = 2

gamma = 0.1
sigma = gamma
omega = 0.005
nu = 3

c = np.array((.2, 0))
c_star = 0.15
r_star = np.array((0.286, 0))
beta_vec = np.array((0.15, 0.19))
beta_star = 0.1691405296549373

def evaluate_average_transmission_rate(r):
    return np.inner(beta_vec, softmax(r-c))

def evaluate_average_cost(r):
    return np.inner(r, softmax(r-c))

## Function call to compute lambda in (20) of the EPG-PBR paper
## Currently, we have a simple (rather naive) searching algorithm
def compute_lambda(beta_bar):
    lambda_parameter_max = 0
    lambda_parameter_min = -1000
    lambda_parameter = (lambda_parameter_min+lambda_parameter_max)/2
    x = softmax(lambda_parameter*beta_vec)
    
    while (np.abs(beta_bar - np.inner(beta_vec, x))>10e-8):
        lambda_parameter = (lambda_parameter_min+lambda_parameter_max)/2
        x = softmax(lambda_parameter*beta_vec)

        if (beta_bar > np.inner(beta_vec, x)):
            lambda_parameter_min = lambda_parameter
        else:
            lambda_parameter_max = lambda_parameter

    # print(np.abs(beta_bar - np.inner(beta_vec, x)))

    return lambda_parameter, x

## Function call computes the cost bound (20) of the EPG-PBR paper
def compute_cost_bound(mu_upper_bound, beta_bar):
    lambda_parameter, x = compute_lambda(beta_bar)
            
    cost_bound = mu_upper_bound*lambda_parameter*(beta_bar-np.max(beta_vec)) + \
        np.inner(c, x)

    return cost_bound

def compute_cost_upper_bounds(mu_upper_bound, beta_bars):
    cost_bounds = []
    
    for beta_bar in beta_bars:
        cost_bound = compute_cost_bound(mu_upper_bound, beta_bar)
        cost_bounds.append(cost_bound)

    return cost_bounds

def compute_beta_bars(mu_upper_bounds):
    beta_bars = []

    for mu_upper_bound in mu_upper_bounds:
        # print('mu_upper_bound: {}'.format(mu_upper_bound))
    
        beta_average = np.sum(beta_vec)/2
        beta_bar_max = beta_average+.0001
        beta_bar_min = np.min(beta_vec)+.0001

        ## Bisection method
        cost_bound = c_star+10
        while (np.abs(cost_bound-c_star) >= 10e-5):
            beta_bar = (beta_bar_max+beta_bar_min)/2
            cost_bound = compute_cost_bound(mu_upper_bound, beta_bar)

            if (cost_bound > c_star):
                beta_bar_min = beta_bar
            else:
                beta_bar_max = beta_bar

        beta_bars.append(beta_bar)

    return beta_bars

if (__name__ == '__main__'):

    mu_upper_bound = 1
    beta_average = np.sum(beta_vec)/2
    beta_bar_max = beta_average+.0001
    # beta_bar_min = np.min(beta_vec)+.0001
    beta_bar_min = 0.16-0.0001
    beta_bars = np.arange(beta_bar_min, beta_bar_max, .0001) 
    cost_bounds = compute_cost_upper_bounds(mu_upper_bound, beta_bars)

    plt.plot(beta_bars, cost_bounds)
    plt.plot(beta_star, c_star, 'rx')
    plt.ylim([-.01, 1.0])
    plt.xlabel('beta_bar')
    plt.ylabel('cost_upper_bound')
    # plt.savefig('/home/shinkyu/data_drive/outputs/epidemic_games/cost_bound_mu_{}.png'.format(mu_upper_bound), format='png')
    #plt.show()

    mu_upper_bounds = np.arange(1, 5, .1)
    beta_bars2 = compute_beta_bars(mu_upper_bounds)

    ## Plot mu_upper_bounds vs beta_bars
    plt.plot(mu_upper_bounds, beta_bars2)
    plt.xlabel('mu_upper_bound')
    plt.ylabel('beta_bars')
    # plt.savefig('/home/shinkyu/data_drive/outputs/epidemic_games/cost_bound_mu_upper_bound.png', format='png')
    #plt.show()


