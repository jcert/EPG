# EPG-PBR

This repository complements the paper titled [Epidemic Population Games And Perturbed Best Response Dynamics](https://doi.org/10.48550/arXiv.2401.15475), and accommodates Julia files for simulating an epidemic compartmental model coupled to a population game.

## Concept

This paper proposes an approach to mitigate epidemic spread in a population of strategic agents by encouraging safer behaviors through carefully designed rewards. These rewards, which vary according to the state of the epidemic, are ascribed by a dynamic payoff mechanism we seek to design. We use a modified SIRS model to track how the epidemic progresses in response to the population's agents strategic choices. By employing perturbed best response evolutionary dynamics to model the population's strategic behavior, we extend previous related work so as to allow for noise in the agents' perceptions of the rewards and intrinsic costs of the available strategies. Central to our approach is the use of system-theoretic methods and passivity concepts to obtain a Lyapunov function, ensuring the global asymptotic stability of an endemic equilibrium with minimized infection prevalence, under budget constraints. We use the Lyapunov function to construct anytime upper bounds for the size of the population's infectious fraction. For a class of one-parameter perturbed best response models, we propose a method to learn the model's parameter from data.


## Requirements
- Julia 1.11.0

## How to use
Following the [guide on environments](https://pkgdocs.julialang.org/v1/), you can open Julia in a terminal, press `]` to access the package manager, type `activate .` and then `instantiate`. 
After installing all the required software you can press backspace to exit the package manager, now you should have all the required libraries to run the code. To run the code either use Jupyter notebook for the interactive plot or open Julia and then type `include("genallfig.jl")` to run the main simulation (that will generate all figures).


## genallfig.jl
Runs the individual code files to generate all figures


## SIRS\_EDM\_PBR\_fullinfo\_dist\_logit.jl
## SIRS\_EDM\_PBR\_fullinfo\_dist\_logit\_kappa.jl
## SIRS\_EDM\_PBR\_fullinfo\_dist\_logit\_kappa.jl
## SIRS\_EDM_PBR\_fullinfo.jl
## SIRS\_EDM_PBR\_fullinfo\_dist.jl
Generates figures made with Julia

## plot\_shinkyu\_1.jl
Run the Python files `compute_cost_bound1.py`, `compute_cost_bound2.py`, and `compute_cost_bound5.py`.
## plot\_shinkyu\_2.jl
Run the Python files `epg_simulation_scenario_1.py` and `epg_simulation_scenario_2.py`.

