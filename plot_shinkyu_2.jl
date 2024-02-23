using Plots
using PyCall
using LaTeXStrings



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
    plt_size = (400,300)
    default(;DEF_CONFIG...)
else
    plt_size = (400,300)
    #plt_size = (900,900)
    default(linewidth = 4, markersize=10, margin = 5*Plots.mm,
        tickfontsize=16, guidefontsize=26, legend_font_pointsize=16)

end

py"""
exec(open("epg_simulation_scenario_1.py").read(), globals(), locals())
"""

T_revise = py"T_revise"

I_trajectory_1, R_trajectory_1 = py"I_trajectory, R_trajectory"
q_trajectory_1, x_trajectory_1 = py"q_trajectory, x_trajectory"
I_star_1 = py"I_star"
time_1 = py"time"
average_reward_1 = py"average_reward"


py"""
exec(open("epg_simulation_scenario_2.py").read(), globals(), locals())
"""

I_trajectory_2, R_trajectory_2 = py"I_trajectory, R_trajectory"
q_trajectory_2, x_trajectory_2 = py"q_trajectory, x_trajectory"
I_star_2 = py"I_star"
time_2 = py"time"
average_reward_2 = py"average_reward"


ds = 10 # downsample rate, 1 is no downsample, 2 is half of the data
plot(legend=:topright)
plot!(time_1[1:ds:end], I_trajectory_1[1:ds:end]/I_star_1, label="Scenario 1")
plot!(time_2[1:ds:end], I_trajectory_2[1:ds:end]/I_star_2, label="Scenario 2")
plot!([T_revise,T_revise],[0.7,1.7],linestyle=:dash, label=nothing)
xlabel!("days")
ylabel!(L"I(t)/I^\ast")
ylims!(0.7, 1.7)
savefig("images/infectious.$(plt_ext)")

plot(legend=:bottomright)
plot!(time_1[1:ds:end], average_reward_1[1:ds:end], label="Scenario 1")
plot!(time_2[1:ds:end], average_reward_2[1:ds:end], label="Scenario 2")
plot!([T_revise,T_revise],[-2.1, 1.1],linestyle=:dash, label=nothing)
ylabel!(L"r'(t)x(t)")
xlabel!("days")
ylims!(-2.1, 1.1)
savefig("images/average_reward.$(plt_ext)")

