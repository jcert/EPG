import pdb, pickle, sys, itertools
import yaml
from yaml.loader import SafeLoader
import argparse
#from loguru import logger
from copy import deepcopy

import numpy as np
from scipy import integrate
from scipy.special import softmax

import matplotlib.pyplot as plt


## Simulation parameters
N_strategy = 2

gamma = 0.1
sigma = gamma
omega = 0.005
upsilon = 3
kappa = 1

c = np.array((.2, 0))
beta_vec = np.array((0.15, 0.19))

r_bar = c
beta_bar = 0.167

c_star = 1
r_star = np.array((1.3248, 0))
beta_star = 0.1598

eta = omega/(omega+gamma)
I_star = eta*(1-sigma/beta_star)

T_revise = 240 ## the planner updates \mu (the choice function parameter) at T_revise
simulation_time = 3000 ## total simulation time (needs to be a multiple of 30)

def epg_dynamics(y,t):
    I = y[0]
    R = y[1]
    q = y[2]
    x = np.array(y[3:])
    
    B = np.inner(x, beta_vec)

    I_hat = eta*(1 - sigma/B)
    R_hat = (1-eta)*(1 - sigma/B)

    dI = (B*(1 - I - R) - sigma)*I
    dR = gamma*I - omega*R

    if (t <= T_revise):
        dq = (I_hat - I) + eta*(np.log(I) - np.log(I_hat)) + upsilon**2*(beta_bar - B) +\
            (B/gamma)*(R - R_hat)*(1 - eta - R)
        
    else:
        dq = (I_hat - I) + eta*(np.log(I) - np.log(I_hat)) + upsilon**2*(beta_star - B) +\
            (B/gamma)*(R - R_hat)*(1 - eta - R)

    r = q*beta_vec + r_bar
    
    dq = kappa*dq
    dx = softmax(r-c) - x

    dy = np.concatenate(([dI, dR, dq], dx))
    
    return dy

def evaluate_lyapunov_function(t, I, R, x_1, q, r_bar):
    x = (x_1, 1-x_1)
    p = q*beta_vec + r_bar - c     
    B = np.inner(x, beta_vec)
    I_hat = eta*(1 - sigma/B)
    R_hat = (1-eta)*(1 - sigma/B)
    y = softmax(p)
    
    lyapunov_function = 0
    if (t <= T_revise):
        lyapunov_function +=  kappa*(B*(I-I_hat) + B*I_hat * np.log(I_hat/I) + \
                                     (1/(2*gamma))*(B*R - B*R_hat)**2 + (upsilon**2/2)*(B-beta_bar)**2)
    else:
        lyapunov_function +=  kappa*(B*(I-I_hat) + B*I_hat * np.log(I_hat/I) + \
                                     (1/(2*gamma))*(B*R - B*R_hat)**2 + (upsilon**2/2)*(B-beta_star)**2)
        
    lyapunov_function += np.inner(y,p)-np.sum(y*np.log(y)) - (np.inner(x,p)-np.sum(x*np.log(x)))

    return lyapunov_function

def evaluate_I_bound(t, alpha):
    ## Brute force evaluation (needs to be improved later...)
    I_range = np.arange(0.01, 1, .01)
    R_range = np.arange(0.01, 1, .01)
    B_range = np.arange(beta_vec[0], beta_vec[1], .01)
    
    bound = 0
    for (I, R, B) in itertools.product(I_range, R_range, B_range):
        I_hat = eta*(1 - sigma/B)
        R_hat = (1-eta)*(1 - sigma/B)
        R = R_hat
        p = q*beta_vec + r_bar - c
        y = softmax(p)

        if (t <= T_revise):
            L_value = kappa*(B*(I-I_hat) + B*I_hat * np.log(I_hat/I) + (1/(2*gamma))*(B*R - B*R_hat)**2 + (upsilon**2/2)*(B-beta_bar)**2)
        else:
            L_value = kappa*(B*(I-I_hat) + B*I_hat * np.log(I_hat/I) + (1/(2*gamma))*(B*R - B*R_hat)**2 + (upsilon**2/2)*(B-beta_star)**2)

        if (L_value <= alpha):
            bound = np.max((bound, I))

    return bound

if (__name__ == '__main__'):

    dt = 0.1
    time = np.arange(0, simulation_time, dt)
    y_initial = (0.0159, 0.318, 0, 0.997, 0.003)

    y_trajectory = []
    average_reward = []
    for i in range(simulation_time//30):
        time_i = np.arange(30*i, 30*(i+1), dt)

        if (len(y_trajectory) == 0):
            y_trajectory = integrate.odeint(epg_dynamics, y_initial, time_i)

            q_trajectory = np.asarray(y_trajectory[:,2])
            x_trajectory = np.asarray(y_trajectory[:,3:])

            average_reward = [np.inner(q_trajectory[j]*beta_vec+r_bar, x_trajectory[j])
                               for j in range(len(time_i))]
            average_reward = np.asarray(average_reward)

        else:
            y_trajectory = np.concatenate((y_trajectory, integrate.odeint(epg_dynamics, y_initial, time_i)))
            
            q_trajectory = np.asarray(y_trajectory[-len(time_i):,2])
            x_trajectory = np.asarray(y_trajectory[-len(time_i):,3:])
            average_reward = np.concatenate((average_reward,
                                            [np.inner(q_trajectory[j]*beta_vec+r_bar, x_trajectory[j])
                                             for j in range(len(time_i))]))

        y_initial = y_trajectory[-1]
        
        if (time_i[-1] > T_revise):
            I_trajectory = y_trajectory[:,0]
            R_trajectory = y_trajectory[:,1]
            q_trajectory = y_trajectory[:,2]
            x_trajectory = y_trajectory[:,3:]
            I = I_trajectory[-1]
            R = R_trajectory[-1]
            q = q_trajectory[-1]
            x_1 = x_trajectory[-1][0]

            ## Find r_bar closest to r_star under which alpha is less than 0.0004
            b_l = 0
            b_u = 1
            for _ in range(100000):
                b = (b_l+b_u)/2
                r_tmp = (1-b)*r_bar+b*r_star
                alpha = evaluate_lyapunov_function(time_i[-1], I, R, x_1, q, r_tmp)

                if (alpha > 0.0004):
                    b_u = b
                else:
                    b_l = b

                if (b_u-b_l < .001):
                    break

            r_bar = r_tmp

            print('b: {}'.format(b))
            print('alpha: {}'.format(alpha))
            # print(evaluate_I_bound(T_revise, alpha)/I_star)

    I_trajectory = y_trajectory[:,0]
    R_trajectory = y_trajectory[:,1]
    q_trajectory = y_trajectory[:,2]
    x_trajectory = y_trajectory[:,3:]
    
    plt.plot(time, I_trajectory/I_star)
    plt.xlabel('days')
    plt.ylabel(r'$I/I^\ast$')
    plt.ylim([0.7, 1.7])
    plt.show()

    plt.plot(time, average_reward)
    plt.xlabel('days')
    plt.ylabel('average_reward')
    plt.ylim([-2.1, 1.1])
    plt.show()
