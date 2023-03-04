using DifferentialEquations
using Plots
using Plotly
using JuMP
using Ipopt
using GLPK
using ProgressBars
using LaTeXStrings   


throw("This file is for a different paper, do not use as it is!")

#plotly()
gr()
include("lib/SIRS_Game.jl")
include("lib/Dynamics.jl")
   

g0 = SIRS_Game(2,fp)

g0.x_star
g0.σ = 0.1
g0.ω = 0.005
g0.γ = g0.σ
g0.υ = 2.0
g0.β   = [0.15;0.19]
g0.c   = [0.2;0.0]
g0.c_star = 0.1
g0.ρ = 0.0

fixall!(g0)

I = Ib(g0,g0.β[1])
R = Rb(g0,g0.β[1])
S = 1.0-I-R

W = [g0.β[1]*I;g0.β[1]*R;1.0;0.0]

## assertions
# betas are in increasing order
@assert all(diff(g0.β).>0)
# c vector is in decreasing order
@assert all(diff(g0.c).<0)
# σ < β[1]
@assert all(g0.σ.<g0.β)
#c_star > g0.c[end]
@assert g0.c_star>g0.c[end]


##
## Figure 1
#solution using nonlinear programming
T = exp.(range(-5,stop=0,length=10))
bounds_list  = []
optimal_list = []
fixall!(g0)
for g0.υ = ProgressBar(T)
    for g0.β_star ∈ [0.16, 0.17, 0.18] 
        β̃  = abs(g0.β[1]-g0.β_star)
        #fixall!(g0)
        α =  0.5*(g0.υ*β̃ )^2
        m = Model(Ipopt.Optimizer)
        set_optimizer_attribute(m, "tol", 1e-12)
        set_optimizer_attribute(m, "acceptable_tol",  1e-12)      
        set_optimizer_attribute(m, "dual_inf_tol",  1e-12)
        set_optimizer_attribute(m, "acceptable_constr_viol_tol",  1e-12)
        set_optimizer_attribute(m, "bound_relax_factor",  0.0)
        set_optimizer_attribute(m, "mu_init",  1e-1)
        set_optimizer_attribute(m, "gamma_theta", 1e-2)
        set_optimizer_attribute(m, "honor_original_bounds",  "yes")
        set_optimizer_attribute(m, "max_iter", 10000)

        
        
        set_silent(m)

        @variable(m, I >= 0)
        @variable(m, R >= 0)
        
        @variable(m, maximum(g0.β) >= B >= minimum(g0.β))
        @constraint(m, R+I<=B)


        @NLconstraint(m, (I-g0.η*(B-g0.σ))+g0.η*(B-g0.σ)*log(g0.η*(B-g0.σ)/I)+(R-(1-g0.η)*(B-g0.σ))^2/(2*g0.γ)+g0.υ^2*(B-g0.β_star)^2/2<=α   )


        @NLobjective(m, Max, I/B )
        optimize!(m)
        up_bd = objective_value(m)/(g0.η*(1-g0.σ/g0.β_star)) 
        
        #println("$(value(B))  $(g0.υ)  ")

        push!(optimal_list, primal_status(m))
#=β_star
        @NLobjective(m, Max, (-I+g0.η*(g0.β_star-g0.σ))/B  )
        optimize!(m)
        lo_bd = objective_value(m) 
=#
        #push!(bounds_list, [β̃ ,g0.υ,max(lo_bd,up_bd)])
        push!(bounds_list, [g0.β_star,g0.υ,max(0,up_bd)])
        push!(optimal_list, primal_status(m))

        if  termination_status(m) != JuMP.MOI.LOCALLY_SOLVED
            print(termination_status(m))
            print(m)
        end

    end
end

bounds_list_aux = bounds_list
bounds_list_aux = permutedims(hcat(bounds_list_aux...))
Plots.plot()
for i in unique(bounds_list_aux[:,1])
    aux = bounds_list_aux[bounds_list_aux[:,1].==i,2:3]
    Plots.plot!( aux[:,1], aux[:,2], label=L"β^*="*"$(round(i,digits=4))", title=" υ vs π* ", xlabel="υ", ylabel=L"\pi_\upsilon^*\left(0.5 \upsilon^2 \tilde{\beta}^2\right)", legend=:outerright) 
end
Plots.plot!() #xaxis=:log,yaxis=:log
#println(optimal_list)

##
Plots.savefig("images/NLP_alpha_vs_pi_star")



#solution using feasibility check of convex programs
##this works too, but it is way slower than the approach above
#=

bounds_list  = []
optimal_list = []

for g0.υ = ProgressBar([0.001,0.01,0.06,0.1,0.3,1.0,10.0,100.0]) 
    fixall!(g0)
    for α = ProgressBar(T)
        l_0 = 1.0
        l_1 = 0.0
        p   = (l_0+l_1)/2 

        m = Model(Ipopt.Optimizer)
        set_silent(m)

        @variable(m, I >= 0)
        @variable(m, g0.β[2] >= B >= g0.β[1])
        @variable(m, R >= 0)


        @constraint(m, I+R<=B)
        @objective(m, Max, 1 )

        @NLconstraint(m, (I-g0.η*(B-g0.σ))+g0.η*(B-g0.σ)*log(g0.η*(B-g0.σ)/I)+(R-(1-g0.η)*(B-g0.σ))^2/(2*g0.γ)+g0.υ*(B-g0.β_star)^2/2<=α   )
        g = p
        myconstr = @constraint(m, g*B<=(-I+g0.η*(g0.β_star-g0.σ)))
        for nnnn=1:10
            g = p
            delete(m, myconstr)
            myconstr = @constraint(m, g*B<=(-I+g0.η*(g0.β_star-g0.σ)))


            optimize!(m)
            #println("$(p)  $(primal_status(m))")
            if (primal_status(m)==MOI.FEASIBLE_POINT)
                l_0 = l_0
                l_1 = p
            else
                l_0 = p
                l_1 = l_1
            end
            p   = (l_0+l_1)/2 
            if l_0-l_1<1e-4
                break
            end
        end
        #println("$(p)  $(primal_status(m))")
        #println("sdfsdfsdf")

        l_0 = 1.0
        l_1 = 0.0
        n   = (l_0+l_1)/2 
        g = n

        for nnnn=1:10
            g = n
            delete(m, myconstr)
            myconstr = @constraint(m, g*B<=(I-g0.η*(g0.β_star-g0.σ)))

            optimize!(m)
            #println("n = $n stat $(primal_status(m))")
            if (primal_status(m)==MOI.FEASIBLE_POINT)
                l_0 = l_0
                l_1 = n
            else
                l_0 = n
                l_1 = l_1
            end
            n   = (l_0+l_1)/2 
            if l_0-l_1<1e-4
                break
            end
        end

        #println("$(n)  $(primal_status(m))")

        push!(bounds_list, [g0.υ ,α,max(p,n)])
    end
end
bounds_list = permutedims(hcat(bounds_list...))
Plots.plot()
for i in unique(bounds_list[:,1])
    aux = bounds_list[bounds_list[:,1].==i,2:3]
    Plots.plot!( aux[:,1], aux[:,2], label="υ=$i", title="α vs π* ", xlabel="α", ylabel="π*") 
end
Plots.plot!()
#println(optimal_list)
Plots.savefig("QuasiCP_alpha_vs_pi_star")



## They should give us the same results
=#
##




