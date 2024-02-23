using Plots

USE_TIKZ = true
ENV["GKSwstype"]="nul" #removes the warnings when running it over SSH


USE_DEFED = @isdefined USE_TIKZ 

#maybe increase linewidth=4 for tikz

if USE_DEFED && (USE_TIKZ)
    global DEF_CONFIG = (linewidth = 4, markersize=10, margin = 6*Plots.mm,
    tickfontsize=12, guidefontsize=14, legend_font_pointsize=14)
else
    global DEF_CONFIG = (linewidth = 4, markersize=10, margin = 13*Plots.mm,
    tickfontsize=20, guidefontsize=28, legend_font_pointsize=24)
end





#fig:exampleLogit
include("SIRS_EDM_PBR_fullinfo_dist_logit.jl")
#fig:exampleKappaLogit
include("SIRS_EDM_PBR_fullinfo_dist_logit_kappa.jl")

#example 3 
include("SIRS_EDM_PBR_fullinfo.jl")

#fig:exampleManyDistributions
include("SIRS_EDM_PBR_fullinfo_dist.jl")

#no longer used
#include("SIRS_EDM_PBR_learn_choice2.jl")
#include("SIRS_EDM_PBR_learn_choice4.jl")

#fig:cost_upper_bound
include("plot_shinkyu_1.jl") # you have to manually change "mu_upper_bound =" to get each of the 3 plots 

print("almost done")
#fig:scenario
include("plot_shinkyu_2.jl")