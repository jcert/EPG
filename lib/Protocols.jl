include("SIRS_Game.jl")
using QuadGK

#this file contains data structures and functions used by the main code 


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
function smith(g::SIRS_Game,x::Vector{},t,i; λ=1.0, τ=0.0 )
    mysum = 0.0
    for j ∈ 1:(g.NS-1)
        mysum +=  x[xi(g,j)]*max(min(λ*(g.F(g,x,t,i)-g.F(g,x,t,j)), τ),0.0)
    end
    mysum += (1.0-sum(x[xi(g,1:(g.NS-1))]) )*max(min(λ*(g.F(g,x,t,i)-g.F(g,x,t,g.NS)), τ),0.0)
    
    for j ∈ 1:g.NS
        mysum += -x[xi(g,i)]*max(min(λ*(g.F(g,x,t,j)-g.F(g,x,t,i)), τ),0.0)
    end
    
    mysum
end




"""
`logit(g::SIRS_Game,x::Vector{Float64},t::Float64,i) -> Float`

logit dynamics. Takes parameters `g` (an SIRS Game), `x` (a vector with the current state, 
for both epidemic and game), `t` (time), and `i` (strategy).


# Examples
```julia
julia> logit(SIRS_Game(2,(g,x,t,j)->[1,0][j],[4.0,1.0],2.0 ,1.0), zeros(10), 1.0, 1)
0.0
```
""" 
function logit(g::SIRS_Game,x::Vector{},t,i; η=1.0)
    mysum = 0
    for j ∈ 1:(g.NS)
        mysum += exp(inv(η)* g.F(g,x,t,j))
        #@show exp(inv(η)* g.F(g,x,t,j)), η
    end
    mysum =  exp(inv(η)*g.F(g,x,t,i))/mysum
    mysum -=  x[xi(g,i)]
    
    mysum
end

function Q_logit(g::SIRS_Game,z::Vector{}; η=1.0)
    η*sum(z[i]*log(z[i]) for i=1:g.NS)

end

function C_logit(g::SIRS_Game,x::Vector{},t,i; η=1.0)
    mysum = 0
    for j ∈ 1:(g.NS)
        mysum += exp(inv(η)* g.F(g,x,t,j))
        #@show exp(inv(η)* g.F(g,x,t,j)), η
    end
    mysum =  exp(inv(η)*g.F(g,x,t,i))/mysum
    
end

function C_logit_payoff(g::SIRS_Game,payoff::Vector{},t,i; η=1.0)
    mysum = 0
    for j ∈ 1:(g.NS)
        mysum += exp(inv(η)* payoff[j])
        #@show exp(inv(η)* g.F(g,x,t,j)), η
    end
    mysum =  exp(inv(η)*payoff[i])/mysum
    
end


function C_dist(g::SIRS_Game,x::Vector{},t,i; dist=Normal())
    #Make sure that the distribution has suport over the reals!

    r = [ g.F(g,x,t,j) for j=1:g.NS]
    #check the terms here
    [ quadgk( x->pdf(dist,x)*prod(cdf(dist,x+r[k]-r[j]) for j=1:g.NS if j!=k), 
                -Inf, Inf, rtol=1e-5)[1] for k=1:g.NS ][i]
    
end

function C_dist(g::SIRS_Game,r,i; dist=Normal())
    #Make sure that the distribution has suport over the reals!

    #check the terms here
    [ quadgk( x->pdf(dist,x)*prod(cdf(dist,x+r[k]-r[j]-g0.c[k]+g0.c[j]) for j=1:g.NS if j!=k), 
                -Inf, Inf, rtol=1e-5)[1] for k=1:g.NS ][i]
    
end

