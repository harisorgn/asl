abstract type abstract_bandit_environment end
abstract type abstract_reward_process end
abstract type abstract_out_process end

struct bandit_environment{T <: abstract_reward_process, Y <: abstract_out_process} <: abstract_bandit_environment

	reward_process::T
	out_process::Y
	rng::AbstractRNG
end

function (env::bandit_environment)(action, cstep, session)

	update_out_process!(env.out_process, action, cstep, session, env.rng)

	return env.reward_process.r_m[cstep, action, session] + env.out_process.r_m[cstep, action, session]
end

struct OU_process <:abstract_reward_process

	n_warmup_steps::Int64
	n_steps::Int64
	n_bandits::Int64
	n_sessions::Int64
	γ_v::Array{Float64, 1}
	μ_v::Array{Float64, 1}
	σ_v::Array{Float64, 1}
	r_m::Array{Float64}						# steps x bandits x sessions

	function OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v)

		rng = MersenneTwister()

		r_m = zeros(n_steps, n_bandits, n_sessions)

		for session = 1 : n_sessions

			r_v = r_0_v

			for warmup_step = 1 : n_warmup_steps

				r_v = r_v + γ_v .* (μ_v - r_v) + σ_v .* rand(rng, Normal(0, 1), n_bandits)

			end

			r_m[1, :, session] = r_v

			for cstep = 2 : n_steps

				r_m[cstep, :, session] = r_m[cstep - 1, :, session] + γ_v .* (μ_v .- r_m[cstep - 1, :, session]) + 
										σ_v .* rand(rng, Normal(0, 1), n_bandits)

			end
		end

		new(n_warmup_steps, n_steps, n_bandits, n_sessions, γ_v, μ_v, σ_v, r_m)
	end
end

initialise_new_instance(env::bandit_environment{OU_process}) = bandit_environment(OU_process(env.reward_process.n_warmup_steps,
																							env.reward_process.n_steps,
																							env.reward_process.n_bandits,
																							env.reward_process.n_sessions,
																							zeros(env.reward_process.n_bandits),
																							env.reward_process.γ_v,
																							env.reward_process.μ_v,
																							env.reward_process.σ_v),
																					env.out_process,
																					MersenneTwister())

struct no_out_process <: abstract_out_process

	r_m::Array{Float64}						# steps x bandits x sessions

	no_out_process(n_steps, n_bandits, n_sessions) = new(zeros(n_steps, n_bandits, n_sessions))
end

function update_out_process!(process::no_out_process, action, cstep, session, rng)
end

expected_out(process::no_out_process, n_steps, bandit) = zeros(n_steps)

struct distribution_out_process <:abstract_out_process
	
	outlier_distr_v::Array{Distribution, 1}
	r_m::Array{Float64}						# steps x bandits x sessions

	function distribution_out_process(n_steps, n_bandits, n_sessions, outlier_distr_v)

		rng = MersenneTwister()

		r_m = zeros(n_steps, n_bandits, n_sessions)

		for session = 1 : n_sessions
			for cstep = 1 : n_steps

				r_m[cstep, :, session] = [rand(rng, out_distr) for out_distr in outlier_distr_v]

			end
		end

		new(outlier_distr_v, r_m)
	end
end

function update_out_process!(process::distribution_out_process, action, cstep, session, rng)
end

expected_out(process::distribution_out_process, n_steps, bandit) = fill(mean(process.outlier_distr_v[bandit]), n_steps)

struct frequency_out_process <: abstract_out_process

	out_bandit::Int64
	r::Float64
	p_max::Float64
	η_p::Float64
	decay_p::Float64
	p_m::Array{Float64}					# steps x sessions
	r_m::Array{Float64}					# steps x bandits x sessions

	frequency_out_process(n_steps, n_bandits, n_sessions, out_bandit, r, p_max, η_p, decay_p) = new(out_bandit, 
																									r,
																									p_max, 
																									η_p, 
																									decay_p, 
																									zeros(n_steps + 1, n_sessions), 
																									zeros(n_steps, n_bandits, n_sessions))
end

function update_out_process!(process::frequency_out_process, action, cstep, session, rng)
	
	if (action == process.out_bandit)

		 process.p_m[cstep + 1, session] = process.p_m[cstep, session] + process.η_p * (process.p_max - process.p_m[cstep, session])

	else

		process.p_m[cstep + 1, session] = (1.0 - process.decay_p) * process.p_m[cstep, session]

	end

	if rand(rng) < process.p_m[cstep + 1, session]

		process.r_m[cstep, :, session] .= process.r

	end
end

function expected_out(process::frequency_out_process, n_steps, bandit)

	if bandit == process.out_bandit

		p_v = [sum([process.p_max * process.η_p * (1.0 - process.η_p)^(n - i) for i = 1 : n]) for n = 1 : n_steps]

		return p_v .* process.r

	else
		return zeros(n_steps)
	end
end

struct delay_out_process <: abstract_out_process

	out_bandit::Int64
	n_steps::Int64
	r::Float64
	delay_distr::Distribution
	r_m::Array{Float64}					# steps x bandits x sessions

	delay_out_process(n_steps, n_bandits, n_sessions, out_bandit, r, delay_distr) = new(out_bandit,
																						n_steps,  
																						r, 
																						delay_distr, 
																						zeros(n_steps, n_bandits, n_sessions))
end

function update_out_process!(process::delay_out_process, action, cstep, session, rng)
	

	if action == process.out_bandit

		delayed_step = rand(rng, process.delay_distr) + 1 	# when env.outlier_delay_distr is Geometric 
															# number of failures of outlier occurence is being sampled 
															# so +1 is required

		if cstep + delayed_step <= process.n_steps

			process.r_m[cstep + delayed_step, :, session] .= process.r

		end
	end	
end

function expected_out(process::delay_out_process, n_steps, bandit)

	if bandit == process.out_bandit

		return fill(process.r * mean(process.delay_distr) / n_steps, n_steps)

	else
		return zeros(n_steps)
	end
end

struct test_out_process <: abstract_out_process
	
	out_bandit::Int64
	r::Float64
	r_m::Array{Float64}					# steps x bandits x sessions

	test_out_process(n_steps, n_bandits, n_sessions, out_bandit, r) = new(r, 
																		out_bandit, 
																		zeros(n_steps, n_bandits, n_sessions))
end

function update_out_process!(process::test_out_process, action, cstep, session, rng)
	
	if cstep == process.out_bandit

		process.r_m[cstep, action, session] = process.r

	end	
end

function reset_environment!(env::bandit_environment{T, Y}) where {T <: abstract_reward_process, Y <: Union{distribution_out_process, no_out_process}}
end

function reset_environment!(env::bandit_environment{T, Y}) where {T <: abstract_reward_process, Y <: Union{delay_out_process, test_out_process}}

	env.out_process.r_m[:] = zeros(env.reward_process.n_steps, env.reward_process.n_bandits, env.reward_process.n_sessions)
end

function reset_environment!(env::bandit_environment{T, frequency_out_process}) where {T <: abstract_reward_process}

	env.out_process.r_m[:] = zeros(env.reward_process.n_steps, env.reward_process.n_bandits, env.reward_process.n_sessions)
	env.out_process.p_m[:] = zeros(env.reward_process.n_steps + 1, env.reward_process.n_sessions)
end

function run_environment!(env::bandit_environment, agent::abstract_bandit_agent)

	available_action_v = 1 : env.reward_process.n_bandits

	r_accum = 0.0

	for session = 1 : env.reward_process.n_sessions

		agent.latest_action = available_action_v[agent.policy(zeros(agent.n_actions))]

		r = env(agent.latest_action, 1, session)

		r_accum += r

		for cstep = 2 : env.reward_process.n_steps

			action = agent(r, available_action_v)

			r = env(action, cstep, session)

			r_accum += r
		end

		agent.offline()
	end	

	#reset_environment!(env)

	return r_accum
end

function run_environment!(env::bandit_environment, agent::abstract_optimal_bandit_agent)

	available_action_v = 1 : env.reward_process.n_bandits

	for session = 1 : env.reward_process.n_sessions
		
		r = 0.0

		for cstep = 1 : env.reward_process.n_steps

			action = agent(env.reward_process.r_m[cstep, :, session] + env.out_process.r_m[cstep, :, session], 
							available_action_v)

			r = env(action, cstep, session)
		end
	end	

	reset_environment!(env)
end

