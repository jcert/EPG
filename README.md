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
Jupyter Notebook file with an interactive plot, in it you can change the parameters of the game and epidemic disease.

This simulation follows the structure of example 1, in which the system starts from an equilibrium and, due to the dynamic payoff, goes to another equilibrium. In the paper, we consider the case in which expensive measures were previously ($t < 0$) in place, but a planner seeks from $t = 0$ onward to relax those measures to reduce the normalized cost rate from $r'(0)x(0) = 0.2$ to a long-term limit of $c^*=0.1$, but in the interactive simulation other initial conditions are allowed (such as starting in an equilibrium associated with a smaller budget and moving to a more expensive one.)

Here are a few screenshots of that 


![iter_sim_run1](https://user-images.githubusercontent.com/13306869/185970707-23018966-b530-4179-b2ae-586966643618.png)
![iter_sim_run2](https://user-images.githubusercontent.com/13306869/185970709-e8f4e701-3744-4dc4-b59f-55465587e6bd.png)
![iter_sim_run3](https://user-images.githubusercontent.com/13306869/185970711-c53a4d83-3e54-47e8-a3ba-e82beba53c59.png)
![iter_sim_run4](https://user-images.githubusercontent.com/13306869/185970715-6dc726d7-a49b-4c4e-bf4f-4c4c81309b11.png)
![iter_sim_run5](https://user-images.githubusercontent.com/13306869/185970719-2bbd1b0d-8a30-4c71-bc21-d2509e4b53e3.png)



