include("SIRS_Game.jl")
include("Protocols.jl")


#this file contains data structures and functions used by the main code 

"""
`fp(g,x,t,j) -> Float`
payoff for the different strategies. Particular to this code


# Examples
```julia
julia> g = SIRS_Game(2,(g,x,t,j)->[1,0][j],[4.0,1.0],2.0 ,1.0)
julia> g.ζ .= [8.0,6.0] 
julia> g.ϑ .= [10.0,4.0]
julia> g.θ .= [1.0,8.0] 
julia> g.ν .= [0.5,3.0] 
julia> g.ξ .= [1.0,1.0] 
julia> fp(g,repeat([0.9,0.1,0.0,0.5,0.0],2),0.0,2)
-10.2
julia> fp(g,repeat([0.9,0.1,0.0,0.5,0.0],2),0.0,2)
-6.859999999999999
```
""" 
function fp(g,x,t,j)
    (g.β*x[qi(g,1)]+g.r_star-g.c)[j]#(g0.β*x[qi(g,1)])[j] #+[-1.0 0; 0.0 -1.0]*x[3:4]+[0.01;.999])[j]
end


"""
`h_smith!(du,u,p,t) -> Array{Float}`

total dynamics of our model, uses smith(p,u,t,i;λ=0.1,τ=0.1) as the protocol 

""" 
function h_smith!(du,u,p,t;kappa=1.0)
    """
    the epidemic state is scaled by B, that is, I := ratio of infected * B 
    """

    for i ∈ 1:(p.NS-1)
        # game dynamic
        du[xi(p,i)] = smith(p,u,t,i;λ=0.1,τ=0.1)
    end


    #make sure that x>=0 and ones(3)'*x == 1 
    x = u[xi(p,1:(p.NS-1))]
    x = [x;1.0-sum(x)]
    x = max.(x,0.0)
    x = x/sum(x)

    if any(u[xi(p,1:(p.NS-1))].>1) || any(u[xi(p,1:(p.NS-1))] .<0)
        u[xi(p,1:(p.NS-1))] .= x[1:(end-1)]
    end


    B = p.β'*x
    
    I = max(u[1],0.0)
    R = max(u[2],0.0)
    if u[1]<0
        u[1] = I
    end
    if u[2]<0
        u[2] = R
    end
    
    I_hat = p.η*(B-p.σ)
    R_hat = (1-p.η)*(B-p.σ)

    g = (I_hat-I)/B+p.η*log(I/I_hat)+(R-R_hat)*(1-p.η-R/B)/p.γ+p.υ^2*(p.β_star-B)
    #g = p.η*log(I/I_hat)*(1-p.η)*(R-R_hat)/p.γ+p.ν*(p.β_star-B)
    #println(I,"    ",I_hat,"    ",p.β,"    ",u[3:4])

    # payoff dynamics
    du[qi(p,1)] = kappa*g

    x_dot = du[xi(p,1:(p.NS-1))]
    x_dot = [x_dot; -sum(x_dot)]
    B_dot = x_dot'*p.β

    # epidemic dynamic
    du[1] = I*(I_hat+R_hat-I-R)+B_dot*I/B
    du[2] = p.ω*(R_hat-R)-p.γ*(I_hat-I)+B_dot*R/B
end



"""
`h_logit!(du,u,p,t) -> Array{Float}`

total dynamics of our model, uses logit(p,u,t,i;η=100.01) as the protocol 

""" 
function h_logit!(du,u,p,t;η=0.5,α=1.0,kappa=1.0)
    """
    the epidemic state is scaled by B, that is, I := ratio of infected * B 
    """

    for i ∈ 1:(p.NS-1)
        # game dynamic
        du[xi(p,i)] = α*logit(p,u,t,i;η=η)
    end


    #make sure that x>=0 and ones(3)'*x == 1 
    x = u[xi(p,1:(p.NS-1))]
    x = [x;1.0-sum(x)]
    x = max.(x,0.0)
    x = x/sum(x)

    if any(u[xi(p,1:(p.NS-1))].>1) || any(u[xi(p,1:(p.NS-1))] .<0)
        u[xi(p,1:(p.NS-1))] .= x[1:(end-1)]
    end


    B = p.β'*x
    
    I = max(u[1],0.0)
    R = max(u[2],0.0)
    if u[1]<0
        u[1] = I
    end
    if u[2]<0
        u[2] = R
    end
    
    I_hat = p.η*(B-p.σ)
    R_hat = (1-p.η)*(B-p.σ)

    g = (I_hat-I)/B+p.η*log(I/I_hat)+(R-R_hat)*(1-p.η-R/B)/p.γ+p.υ^2*(p.β_star-B)
    #g = p.η*log(I/I_hat)*(1-p.η)*(R-R_hat)/p.γ+p.ν*(p.β_star-B)
    #println(I,"    ",I_hat,"    ",p.β,"    ",u[3:4])

    # payoff dynamics
    du[qi(p,1)] = kappa*g

    x_dot = du[xi(p,1:(p.NS-1))]
    x_dot = [x_dot; -sum(x_dot)]
    B_dot = x_dot'*p.β

    # epidemic dynamic
    du[1] = I*(I_hat+R_hat-I-R)+B_dot*I/B
    du[2] = p.ω*(R_hat-R)-p.γ*(I_hat-I)+B_dot*R/B
end





"""
`h_dist!(du,u,p,t) -> Array{Float}`

total dynamics of our model, uses C_dist as the choice function 

"""
function h_dist!(du,u,p,t; dist=Normal(), α=1.0, kappa=1.0)
   """
    the epidemic state is scaled by B, that is, I := ratio of infected * B 
    """
    
    # game dynamic
    du[xi(p,1:(p.NS-1))] = α.*( C_dist(p,u,t,1:(p.NS-1),dist=dist)-u[xi(p, 1:(p.NS-1))] )
    

    #make sure that x>=0 and ones(3)'*x == 1 
    x = u[xi(p,1:(p.NS-1))]
    x = [x;1.0-sum(x)]
    x = max.(x,0.0)
    x = x/sum(x)

    if any(u[xi(p,1:(p.NS-1))].>1) || any(u[xi(p,1:(p.NS-1))] .<0)
        u[xi(p,1:(p.NS-1))] .= x[1:(end-1)]
    end


    B = p.β'*x
    
    I = max(u[1],0.0)
    R = max(u[2],0.0)
    if u[1]<0
        u[1] = I
    end
    if u[2]<0
        u[2] = R
    end
    
    I_hat = p.η*(B-p.σ)
    R_hat = (1-p.η)*(B-p.σ)

    g = (I_hat-I)/B+p.η*log(I/I_hat)+(R-R_hat)*(1-p.η-R/B)/p.γ+p.υ^2*(p.β_star-B)
    #g = p.η*log(I/I_hat)*(1-p.η)*(R-R_hat)/p.γ+p.ν*(p.β_star-B)
    #println(I,"    ",I_hat,"    ",p.β,"    ",u[3:4])

    # payoff dynamics
    du[qi(p,1)] = kappa*g

    x_dot = du[xi(p,1:(p.NS-1))]
    x_dot = [x_dot; -sum(x_dot)]
    B_dot = x_dot'*p.β

    # epidemic dynamic
    du[1] = I*(I_hat+R_hat-I-R)+B_dot*I/B
    du[2] = p.ω*(R_hat-R)-p.γ*(I_hat-I)+B_dot*R/B
end