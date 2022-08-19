# EPG

This repository complements the paper titled "Epidemic Population Games And Evolutionary Dynamics", and accommodates Julia files for simulating an epidemic compartmental model coupled to a population game.

## Concept
We propose a system theoretic approach to select and stabilize the endemic equilibrium of an SIRS epidemic model in which the decisions of a population of strategically interacting agents determine the transmission rate. Specifically, the population's agents recurrently revise their choices out of a set of strategies that impact to varying levels the transmission rate. A payoff vector quantifying the incentives provided by a planner for each strategy, after deducting the strategies' intrinsic costs, influences the revision process. An evolutionary dynamics model captures the population's preferences in the revision process by specifying as a function of the payoff vector the rates at which the agents' choices flow toward strategies with higher payoffs. Our main result is a dynamic payoff mechanism that is guaranteed to steer the epidemic variables (via incentives to the population) to the endemic equilibrium with the smallest infectious fraction, subject to cost constraints. We use a Lyapunov function not only to establish convergence but also to obtain an (anytime) upper bound for the peak size of the population's infectious portion. 

[preprint](https://arxiv.org/abs/2201.10529)

[![DOI](https://zenodo.org/badge/395116198.svg)](https://zenodo.org/badge/latestdoi/395116198)



## Requirements
- Julia 1.6
- (*optional*) jupyter-notebook 6.2

## How to use
Following the [guide on environments](https://pkgdocs.julialang.org/v1.2/environments/), you can open Julia in a terminal, press `]` to access the package manager, type `activate .` and then `instantiate`. 
After installing all the required software you can press backspace to exit the package manager, now you should have all the required libraries to run the code. To run the code either use Jupyter notebook for the interactive plot or open Julia and then type `include("SIRS\_EDM.jl")` to run the main simulation (that will generate Figure 2).

To generate Figure 1, open Julia and run `include("optimizer.jl")`.


## optimizer.jl
File to generate the plot used for Figure 1

## SIRS\_EDM.jl
File to generate the plots used for Figure 2

## SIRS\_EDM.ipybn
Jupyter Notebook file with an interactive plot, in it you can change the parameters of the game and epidemic disease. Here are a few screenshoots of that 

![iter_sim_run1](https://user-images.githubusercontent.com/13306869/182480097-9fe10b72-3d3c-4970-95d8-1dc1d81d2f89.png)
![iter_sim_run2](https://user-images.githubusercontent.com/13306869/182480095-26046b13-0d7c-4344-bb88-0f46580b0a6e.png)
![iter_sim_run3](https://user-images.githubusercontent.com/13306869/182480094-2fe525e6-f7a9-44e7-b26d-202e372cb788.png)
![iter_sim_run4](https://user-images.githubusercontent.com/13306869/182480099-9b8f07ab-5282-47c5-b469-f19da81b7f14.png)






