using Distributions
using Random
using Statistics
#using JLD
#using NLopt

include("agents.jl")
include("environments.jl")
include("auxiliary.jl")

function obj(x::Vector, grad::Vector, env_v, obj_param_v, offline_learning, policy)

	obj = 0.0

	for env in env_v

		n_bias_steps = Int(floor(0.5*env.reward_process.n_steps))
		bias_buffer_length = Int(floor(0.2*env.reward_process.n_steps))

		η_offline = x[1]
		decay_offline = obj_param_v[1]

		η_r = x[2]
		decay_r = obj_param_v[2]

		ε = x[3]

		agent = delta_agent(env.reward_process.n_bandits, η_r, decay_r,
							offline_learning(n_bias_steps, env.reward_process.n_bandits, 
											bias_buffer_length, η_offline, decay_offline), 
							policy(ε))

		obj += run_environment!(env, agent)

	end

	return obj
end

function main()
	
	n_warmup_steps = 0
	n_steps = 40
	n_sessions = 10
	n_bandits = 2

	r_out = 10.0

	μ_v = [3.0, 3.0]
	σ_v = [1.0, 1.0]

	r_0_v = μ_v

	γ_v = [0.01, 0.01]

	env = bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v),
							initialise_delay_out_process(n_steps, n_bandits, n_sessions, r_out), 
							MersenneTwister())

	agent = delta_agent(n_bandits, 0.1, 0.01, 
						offline_bias(20, n_bandits, 10, 0.01, 0.01),
						ε_greedy_policy(0.1))
	
	println(run_environment!(env, agent))

end	

main()


