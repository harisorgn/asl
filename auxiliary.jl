
function get_optimal_agent(env::bandit_environment{T, Y}) where {T <: abstract_reward_process, Y <: abstract_out_process}

	return optimal_bandit_agent()
end

OU_distr(x, t, γ, σ; t_0  = 0.0, x_0 = 0.0) = sqrt(γ / (2.0 * pi * (σ^2.0 / 2.0) * (1.0 - exp(-2.0 * γ * (t - t_0))))) * 
											exp(-(γ * (x - x_0 * exp(-γ * (t - t_0)))^2.0)/
												(2.0 * (σ^2.0 / 2.0) * (1.0 - exp(-2.0 * γ * (t - t_0)))))

function initialise_OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)

	r_0_v = zeros(n_bandits)

	return OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v)
end

function initialise_distribution_out_process(n_steps, n_bandits, n_sessions, r_out)

	outlier1_distr = MixtureModel(Normal, [(0.0, 0.01), (r_out, 0.1), (-r_out, 0.1)], 
								  			[0.8, 		0.15, 		0.05])

	outlier2_distr = MixtureModel(Normal, [(0.0, 0.01), (r_out, 0.1), (-r_out, 0.1)], 
								  			[0.9, 		0.05, 		0.05])

	return distribution_out_process(n_steps, n_bandits, n_sessions, [outlier1_distr, outlier2_distr])
end

function initialise_frequency_out_process(n_steps, n_bandits, n_sessions, r_out)
	
	out_bandit = 1
	p_out_max = 0.2 
	η_p_out = 0.05 
	decay_p_out = 0.1

	return frequency_out_process(n_steps, n_bandits, n_sessions, out_bandit, r_out, p_out_max, η_p_out, decay_p_out)
end

function initialise_delay_out_process(n_steps, n_bandits, n_sessions, r_out)

	out_bandit = 1
	out_delay_distr = Geometric(0.1)

	return delay_out_process(n_steps, n_bandits, n_sessions, out_bandit, r_out, out_delay_distr)
end

initialise_test_out_process(n_steps, n_bandits, n_sessions, out_bandit, r_out) = test_out_process(n_steps, n_bandits, n_sessions, out_bandit, r_out)

function run_bandit_example(env::abstract_bandit_environment, agent_v::Array{T, 1}; plot_flag = false) where T <:abstract_bandit_agent

	random_agent = delta_agent(env.reward_process.n_steps, env.reward_process.n_bandits, env.reward_process.n_sessions, 
							0.0, 0.0, 
							offline_bias(env.reward_process.n_steps, env.reward_process.n_bandits, env.reward_process.n_sessions, 
										env.reward_process.n_steps, 0.0, 0.0), 
							ε_greedy_policy(1.0))

	run_environment!(env, random_agent)

	score_v = zeros(length(agent_v))

	c = 1
	for agent in agent_v
	
		run_environment!(env, agent)

		score_v[c] = sum(agent.accumulated_r_v) - sum(random_agent.accumulated_r_v)

		if plot_flag

			println(typeof(agent), " score : ", score_v[c])

			plot_bandit_results(env, agent, 1:10)
		end

		c += 1
	end

	return score_v
end

function run_opt(env_v::Array{T,1}) where T <: abstract_bandit_environment

	#___objective function patameter vector = [decay_offline, decay_reward, γ_ΟU, σ_OU]___
	obj_param_v = [0.01, 0.01, env_v[1].reward_process.γ_v[1], env_v[1].reward_process.σ_v[1]]

	bias_v = run_delta_agent_opt(env_v, obj_param_v, offline_bias, ε_greedy_policy)

	Q_v = run_delta_agent_opt(env_v, obj_param_v, offline_Q, ε_greedy_policy)

	no_offline_v = run_delta_agent_no_offline_opt(env_v, obj_param_v, ε_greedy_policy)


	bias_agent = delta_agent(0, 0, 0, bias_v[2], obj_param_v[2], 
								offline_bias(0, 0, 0, 0, bias_v[1], obj_param_v[1]),
								ε_greedy_policy(bias_v[3]))

	Q_agent = delta_agent(0, 0, 0, Q_v[2], obj_param_v[2], 
							offline_Q(0, 0, 0, 0, Q_v[1], obj_param_v[1]),
							ε_greedy_policy(Q_v[3]))

	no_offline_agent = delta_agent(0, 0, 0, no_offline_v[1], obj_param_v[2], 
									offline_bias(0, 0, 0, 0, 0.0, obj_param_v[1]),
									ε_greedy_policy(no_offline_v[2]))

	
	save(string("opt_res_", rand(MersenneTwister(), 1000:9999), ".jld"), 
		"env_v", env_v, "bias_agent", bias_agent, "Q_agent", Q_agent, "no_offline_agent", no_offline_agent)

end

function run_bandit_test()

	n_warmup_steps = 10
	n_steps = 40
	n_sessions = 10
	n_bandits = 2

	r_0_v = [0.0, 0.0]
	μ_v = [0.0, 0.0]
	σ_v = [1.0, 1.0]
	γ_v = [0.5, 0.5]

	r_out = 10.0

	env = bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, μ_v, σ_v, γ_v),
							test_out_process(n_steps, n_bandits, n_sessions, r_out),
							MersenneTwister())

	(n_steps, n_bandits, n_sessions) = size(env.reward_process.r_m)
	
	n_bias_steps = floor(Int(0.5env.reward_process.n_steps))
	bias_buffer_length = floor(Int(0.2env.reward_process.n_steps))

	η_offline = 0.01
	decay_offline = 0.01

	η_r = 0.1
	decay_r = 0.01

	ε = 0.1

	agent = delta_agent(n_steps, n_bandits, n_sessions, 
						η_r, decay_r, 
						offline_Q(n_bias_steps, n_bandits, n_sessions, bias_buffer_length, η_offline, decay_offline), 
						ε_greedy_policy(ε))

	run_bandit_example(env, [agent] ; plot_flag = true)

end
