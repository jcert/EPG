
#this file contains data structures and functions used by the main code 


mutable struct SIRS_Game
    NS::Int             #number of strategies
    XS::Vector{Float64} #strategy state
    XF::Vector{Float64} #fatigue state
    #epidemics state
    S::Float64
    I::Float64
    R::Float64
    #parameters of the epidemics
    β #this can be a function
    γ::Float64
    α::Float64
    #payoff function
    F
    #parameters of the payoff - particular to our payoff structure
    ζ::Vector{Float64} 
    ϑ::Vector{Float64} 
    θ::Vector{Float64} 
    ν::Vector{Float64} 
    ξ::Vector{Float64} 
    function SIRS_Game(ns::Int, f, b, g ,a) 
        F  = f
        NS = ns
        β  = b 
        γ  = g
        α  = a
        XS = zeros(NS)
        XF = zeros(NS)
        ζ  = zeros(NS)
        ϑ  = zeros(NS)
        θ  = zeros(NS)
        ν  = zeros(NS) 
        ξ  = zeros(NS)
        new(NS,XS,XF,1.0,0.0,0.0,β,γ,α,F,ζ,ϑ,θ,ν,ξ)
    end
end


"""
`r(strategy,index::Int) -> Int`

# Examples
```julia
julia> r(2,1)
6
```
""" 
function r(strategy,index::Int)
    5*(strategy-1)+index
end

"""
`r(strategy,index::String) -> Int`

# Examples
```julia
julia> r(3,"X")
14
```
""" 
function r(strategy,index::String)
    if(index=="S" || index=="s")  
        5*(strategy-1)+1
    end
    if(index=="I" || index=="i")  
        5*(strategy-1)+2
    end
    if(index=="R" || index=="r")  
        5*(strategy-1)+3
    end
    if(index=="X" || index=="x")  
        5*(strategy-1)+4
    end
    if(index=="F" || index=="f")  
        5*(strategy-1)+5
    end
end


"""
`smith(g::SIRS_Game,x::Vector{Float64},t::Float64,i) -> Float`

Smith dynamics. Takes parameters `g` (an SIRS Game), `x` (a vector with the current state, 
for both epidemic and game), `t` (time), and `i` (strategy).


# Examples
```julia
julia> smith(SIRS_Game(2,(g,x,t,j)->[1,0][j],[4.0,1.0],2.0 ,1.0), zeros(10), 1.0, 1)
0.0
```
""" 
function smith(g::SIRS_Game,x::Vector{Float64},t::Float64,i)
    sum = 0
    for j ∈ 1:g.NS
        sum +=  x[r(j,4)]*max(g.F(g,x,t,i)-g.F(g,x,t,j),0)
    end
    for j ∈ 1:g.NS
        sum += -x[r(i,4)]*max(g.F(g,x,t,j)-g.F(g,x,t,i),0)
    end
    
    sum
end



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
    if j ∈ 1:g.NS
        I_dot = g.β[j]*x[r(j,1)]*x[r(j,2)]-g.γ*x[r(j,2)] #for now, replace with the I_dot term
        -1*(g.ζ[j]+g.ϑ[j]*I_dot+g.θ[j]*x[r(j,2)]+g.ν[j]*x[r(j,5)]+g.ξ[j]*x[r(j,4)])
    end
end


"""
`h!(du,u,p,t) -> Array{Float}`

total dynamics of our model

""" 
function h!(du,u,p,t)
   
    #dynamic per strategy
    for i ∈ 1:p.NS
        # epidemic dynamic
        du[r(i,1)] = -u[r(i,1)]*sum([p.β[j]*u[r(j,4)]*u[r(j,2)] for j ∈ 1:p.NS]) + p.α*u[r(i,3)]
        du[r(i,2)] =  u[r(i,1)]*sum([p.β[j]*u[r(j,4)]*u[r(j,2)] for j ∈ 1:p.NS]) - p.γ*u[r(i,2)]
        du[r(i,3)] =  p.γ*u[r(i,2)] - p.α*u[r(i,3)]
        # game dynamic
        du[r(i,4)] = smith(p,u,t,i)
        # fatigue dynamics
        du[r(i,5)] = 0.03*(u[r(i,4)] - u[r(i,5)])
    end

end
