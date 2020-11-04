
function obj_func_delta(x::Vector, grad::Vector, env_v, obj_param_v)

	obj = 0.0

	for env in env_v

		(n_steps, n_bandits, n_sessions) = size(env.r_m)

		n_bias_steps = Int(floor(0.5*env.n_steps))
		bias_buffer_length = Int(floor(0.2*env.n_steps))

		η_b = x[1]
		decay_b = 0.01

		η_r_delta = x[2]
		decay_r = 0.01

		ε = x[3]

		agent = delta_agent(n_steps, n_bandits, n_sessions, η_r_delta, decay_r,
							offline_bias(n_bias_steps, n_bandits, n_sessions, bias_buffer_length, η_b, decay_b), 
							ε_greedy_policy(ε))

		run_environment!(env, agent)

		obj += sum(agent.accumulated_r_v)

	end

	return obj
end

function obj_func_prob_delta(x::Vector, grad::Vector, env_v, obj_param_v)
	
	obj = 0.0

	for env in env_v

		(n_steps, n_bandits, n_sessions) = size(env.r_m)
	
		n_bias_steps = Int(floor(0.5*env.n_steps))
		bias_buffer_length = Int(floor(0.2*env.n_steps))

		η_b = x[1]
		decay_b = 0.01

		μ_prob_delta = x[3]
		σ_prob_delta = x[4]
		decay_r = 0.01

		η_r_prob_delta = x[2]

		ε = x[5]

		agent = probabilistic_delta_agent(n_steps, n_bandits, n_sessions, 
										η_r_prob_delta, decay_r, μ_prob_delta, σ_prob_delta, 
										offline_bias(n_bias_steps, n_bandits, n_sessions, bias_buffer_length, η_b, decay_b), 
										ε_greedy_policy(ε))

		run_environment!(env, agent)

		obj += sum(agent.accumulated_r_v)

	end

	return obj
end

constraint_prob_delta(x::Vector, grad::Vector) = x[2] - 1.0/pdf(Normal(x[3], x[4]), x[3])

function run_delta_agent_opt(env_v::Array{T,1}, obj_param_v) where T <: abstract_bandit_environment

	# vector to be optimised = [η_b, η_r, ε]

	opt = Opt(:LN_COBYLA, 3)
	opt.lower_bounds = [0.0, 0.0, 0.0]
	opt.upper_bounds = [1.0, 1.0, 1.0]
	opt.ftol_rel = 1e-16
	opt.ftol_abs = 1e-16
	opt.xtol_rel = 1e-16
	opt.xtol_abs = 1e-16

	opt.max_objective = (x, g) -> obj_func_delta(x, g, env_v, obj_param_v)

	max_opt = -999.0
	max_x_v = zeros(length(opt.lower_bounds))

	for η_b_0 in 0.01:0.3:1.0
		for η_r_0 in 0.1:0.3:1.0
			for ε_0 in 0.0:0.3:1.0

				(max_f, max_x, ret) = optimize(opt, [η_b_0, η_r_0, ε_0])

				if max_f > max_opt
					max_x_v[:] = max_x
					max_opt = max_f
				end
			end
		end
	end

	println("got $max_opt at $max_x_v")

	return max_x_v
end

function run_prob_delta_agent_opt(env_v::Array{T,1}, obj_param_v) where T <: abstract_bandit_environment

	# vector to be optimised = [η_b, η_r, μ, σ, ε]

	opt = Opt(:LN_COBYLA, 5)
	opt.lower_bounds = [0.0, 0.0, -Inf, 0.0, 0.0]
	opt.upper_bounds = [1.0, Inf, Inf, Inf, 1.0]
	opt.ftol_rel = 1e-16
	opt.ftol_abs = 1e-16
	opt.xtol_rel = 1e-16
	opt.xtol_abs = 1e-16

	opt.max_objective = (x, g) -> obj_func_prob_delta(x, g, env_v, obj_param_v)

	inequality_constraint!(opt, constraint_prob_delta, 1e-8)

	max_opt = -999.0
	max_x_v = zeros(length(opt.lower_bounds))

	for η_b_0 in 0.01:0.3:1.0
		for μ_0 in -0.5:0.25:0.5
			for σ_0 in 0.1:0.3:1.0
				for ε_0 in 0.0:0.3:1.0
					for η_r_0 in range(0.1, stop = (0.8/pdf(Normal(μ_0, σ_0), μ_0)), length = 5)

						(max_f, max_x, ret) = optimize(opt, [η_b_0, η_r_0, μ_0, σ_0, ε_0])

						if max_f > max_opt
							max_x_v[:] = max_x
							max_opt = max_f
						end
					end
				end
			end
		end
	end
	
	println("got $max_opt at $max_x_v")

	return max_x_v
end