# EPG

This repository complements the paper titled "Epidemic Population Games And Evolutionary Dynamics", and accommodates Julia files for simulating a epidemic compartmental model coupled to a population game.

## Concept
We propose a system theoretic approach to select and stabilize the endemic equilibrium of an SIRS epidemic model in which the decisions of a population of strategically interacting agents determine the transmission rate. Specifically, the population's agents recurrently revise their choices out of a set of strategies that impact to varying levels the transmission rate. A payoff vector quantifying the incentives provided by a planner for each strategy, after deducting the strategies' intrinsic costs, influences the revision process. An evolutionary dynamics model captures the population's preferences in the revision process by specifying as a function of the payoff vector the rates at which the agents' choices flow toward strategies with higher payoffs. Our main result is a dynamic payoff mechanism that is guaranteed to steer the epidemic variables (via incentives to the population) to the endemic equilibrium with the smallest infectious fraction, subject to cost constraints. We use a Lyapunov function not only to establish convergence but also to obtain an (anytime) upper bound for the peak size of the population's infectious portion. 
[preprint](https://arxiv.org/abs/2201.10529)


## Requirements
- Julia 1.6
- (*optional*) jupyter-notebook 6.2

## How to
following the [guide on environments](https://pkgdocs.julialang.org/v1.2/environments/), you can open Julia in a terminal, press `]` to access the package manager, type `activate .` and then `instantiate`. 
After installing all the required software you can press backspace to exit the package manager, now you should have all the required libraries to run the code. To run the code either use Jupyter notebook for the interactive plot or open Julia and then type `include("SIRS\_EDM.jl")` to run the main simulation (that will generate Figure 2).

To generate Figure 1, open Julia and run `include("optimizer.jl")`.


## SIRS\_EDM.jl
File to generate the plots used for Figure 2


## SIRS\_EDM.ipybn
Jupyter Notebook file with an interactive plot, in it you can change the parameters of the game and epidemic disease


## optimizer.jl
File to generate the plot used for Figure 1

