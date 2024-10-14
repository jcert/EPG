using Plots
using PyCall
using LaTeXStrings
using Polynomials



USE_DEFED = @isdefined USE_TIKZ 
if USE_DEFED && (USE_TIKZ)
    pgfplotsx()
    plt_ext = "tikz"
else
    #plotly()
    gr()
    plt_ext = "png"
end


if @isdefined DEF_CONFIG
    plt_size = (800,500)
    default(;DEF_CONFIG...)
else
    plt_size = (800,500)
    #plt_size = (900,900)
    default(linewidth = 3, markersize=10, margin = 25*Plots.mm,
    tickfontsize=12, guidefontsize=20, legend_font_pointsize=18)

end


for i in [2,5]
    py"""
    exec(open("compute_cost_bound"+str($i)+".py").read(), globals(), locals())
    """

    beta_bars, cost_bounds = py"beta_bars, cost_bounds"
    beta_star, c_star = py"beta_star, c_star"
    mu_upper_bounds, beta_bars2 = py"mu_upper_bounds, beta_bars2"


    plot(beta_bars, cost_bounds, legend=nothing, size=plt_size)
    ylabel!("")
    xlabel!(L"\bar{\beta}")
    scatter!([beta_star], [c_star], markershape = :circle)
    ylims!(0.0,1.0)
    savefig("images/cost_bound_mu_$(py"mu_upper_bound").$(plt_ext)")

end

py"""
exec(open("compute_cost_bound1.py").read(), globals(), locals())
"""

beta_bars, cost_bounds = py"beta_bars, cost_bounds"
beta_star, c_star = py"beta_star, c_star"
mu_upper_bounds, beta_bars2 = py"mu_upper_bounds, beta_bars2"


plot(beta_bars, cost_bounds, legend=nothing, size=plt_size)
ylabel!("cost upper bound (23)")
xlabel!(L"\bar{\beta}")
scatter!([beta_star], [c_star], markershape = :circle)
ylims!(0.0,1.0)
savefig("images/cost_bound_mu_$(py"mu_upper_bound").$(plt_ext)")


py"""
exec(open("compute_cost_bound1.py").read(), globals(), locals())
"""

mu_upper_bounds, beta_bars2 = py"mu_upper_bounds, beta_bars2"


plot(mu_upper_bounds, beta_bars2, legend=nothing, size=plt_size)
xlabel!(L"\mu_U")
ylabel!(L"$\bar \beta_{\text{min}}$")
savefig("images/beta_bar_mu_upper_bound_plot.$(plt_ext)")


#f = Polynomials.fit(mu_upper_bounds, beta_bars2,5)
#plot!(1:0.01:5,f.(1:0.01:5))