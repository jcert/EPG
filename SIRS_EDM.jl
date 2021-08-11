using DifferentialEquations
using Plots
using Plotly

plotly()

#file that contains data structures and functions used by the main code 
include("aux.jl")
   
a = 0.08
g0 = SIRS_Game(2,fp,[4.0*a,(a/2)],a ,a/2)

I = 0.01
R = 0.01
S = 1.0-I-R

W = [S;I;R;1.0;0.0;1.0;0.0;0.0;0.0;0.0]

##
g0.ζ .= [8.0,6.0] 
g0.ϑ .= [10.0,4.0]
g0.θ .= [1.0,8.0] 
g0.ν .= [0.5,3.0] 
g0.ξ .= [1.0,1.0] 


prob = ODEProblem(h!,W ,[0.0,420.0],g0)
sol = solve(prob,Tsit5())
#plot(sol, label=["S" "I" "R" "s1" "s2" "f1" "f2"])
Plots.plot(sol, vars=r.(1:2,2), label="I", linewidth=4, thickness_scaling = 1)
xlims!(0.0,510.0)
xlabel!("")
#xlabel!("Time (day)")
#ylabel!("Infected Population (ratio)")
png("infected_12")

##
Plots.plot(sol, vars=r(2,2), label=false, linewidth=4, thickness_scaling = 1)
xlims!(0.0,510.0)
xlabel!("")
png("infected_2")

##
Plots.plot(sol, vars=r.([1,2]',[2,3]), label=["I" "R"], linewidth=4, thickness_scaling = 1)
xlims!(0.0,510.0)
xlabel!("")
#xlabel!("Time (day)")
#ylabel!("Population (ratio)")

##
png("sir_12")
Plots.plot(sol, vars=r.([1,2]',[2,3]), label=["I" "R"], linewidth=4, thickness_scaling = 1)
xlims!(0.0,510.0)
xlabel!("")
png("sir_12")

##
Plots.plot(sol, vars=[r(1,4),r(2,4),r(1,5),r(2,5)], label=["s1" "s2" "f1" "f2"],linewidth=4, thickness_scaling = 1)
xlims!(0.0,510.0)
xlabel!("")
#xlabel!("Time (day)")
#ylabel!("Population (ratio)")
png("strategies_fatigue")


