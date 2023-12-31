module FitODE
using CSV, DataFrames, Statistics, Distributions, Interpolations, QuadGK,
		DiffEqFlux, DifferentialEquations, Printf, Plots, JLD2, Parameters
export callback, loss, weights, fit_diffeq, refine_fit, refine_fit_bfgs,
			calc_gradient, save_data, load_data, read_data, make_loss_args_all
# re-export from DiffEqFlux
export gradient, mma

# Combines ODE and NODE into single code base, with options to switch
# between ODE and NODE. Also provides switch to allow fitting of of initial
# conditions for extra dummy variable dimensions, ie, more variables in
# dynamics than in data, option opt_dummy_u0

# It may be that optimizing dummy vars sets the value when fitting for
# the initial timesteps and then gets fixed there, causing trap
# in local minimum. With random initial value (not optimized), sometimes
# better or worse but over repeated trials may be better than optimizing
# on first values.

# For lynx and hare data, start by log transform. Thus, linear (affine) ODE
# with "bias" term can be interpreted as LV ODE with extra squared term for
# each "species". Data for n=2 2D lynx and hare fit better by using 3 or more
# dimensions, ie, with extra 1 or 2 dummy variables. For NODE, no simple
# interpretation of terms. For NODE, can also use dummy variables, typically
# n=3 or 4 gives better fit than n=2. Perhaps 2D data sit in 3D manifold??

# Occasionally system will lock into local minimum that is clearly not
# a good fit. Rerun program, which seeds with different initial
# parameters and will typically avoid the same local minimum.

# Goal is to find one good fit. So possible need to do a few
# runs with different initial parameter seeds is not a difficulty.
# Program runs reasonably fast on desktop computer, depending on
# hyerparameters.

# If trouble converging, try varying adm_learn, the learning rate, the 
# solver tolerances, and the solver algorithm. If instabilities, lower
# tolerances and use "stiff" ODE solver

# See comments in FitODE_settings.jl

####################################################################
# colors, see MMAColors.jl in my private modules

mma = [RGB(0.3684,0.50678,0.7098),RGB(0.8807,0.61104,0.14204),
			RGB(0.56018,0.69157,0.19489), RGB(0.92253,0.38563,0.20918)];

####################################################################

load_data_warning = true

# These functions called w/in module, no need to call directly.
# However, can be useful for debugging in interactive session.

# ode_data, u0, tspan, tsteps, ode_data_orig = FitODE.read_data(S);
# dudt, ode!, predict = FitODE.setup_diffeq_func(S);

@with_kw struct loss_args
	u0
	prob				# problem for training data period only
	predict
	ode_data
	ode_data_orig
	tsteps				# steps for training data period only
	w
end

# When fit only to training data subset, then need full time period info
struct all_time
	prob_all			# problem for full data set	
	tsteps_all			# steps for full data set
end

make_loss_args_all(L::loss_args, A::all_time) =
					loss_args(L; prob=A.prob_all, tsteps=A.tsteps_all,
					w=ones(length(L.ode_data[:,1]),length(L.ode_data[1,:])))

function read_data(S)
	# Read data and store in matrix ode_data, row 1 for hare, row 2 for lynx
	df = CSV.read(S.csv_file, DataFrame);
	ode_data = permutedims(Array{Float64}(df[:,2:3]));

	# take log and then normalize by average, see DOI: 10.1111/2041-210X.13606
	ode_data = log.(ode_data)
	ode_data = ode_data .- mean(ode_data)
	
	datasize = length(ode_data[1,:]) # Number of time points
	tspan = (0.0f0, Float64(datasize-1)) # Time range
	tsteps = range(tspan[1], tspan[2], length = datasize) # Split to equal steps

	# fit spline to data and use fitted data to train, gauss smoothing is an option
	if S.use_splines
		ode_data_orig = ode_data	# store original data
		hspline = CubicSplineInterpolation(tsteps,ode_data[1,:])
		lspline = CubicSplineInterpolation(tsteps,ode_data[2,:])
		tsteps = range(tspan[1], tspan[2], length = 1+S.pts*(datasize-1)) # Split to equal steps
		if S.use_gauss_filter
			conv(y, spline, sd) = 
				quadgk(x -> spline(x) * pdf(Normal(0,sd),y-x), tspan[1], tspan[2])[1]
			ode_data = vcat([conv(y,hspline,S.filter_sd) for y=tsteps]',
								[conv(y,lspline,S.filter_sd) for y=tsteps]');
		else
			ode_data = vcat(hspline.(tsteps)',lspline.(tsteps)');
		end
		#plot(tsteps,ode_data[1,:]); plot!(tsteps,ode_data[2,:])
	end
	
	u0 = ode_data[:,1] # Initial condition, first time point in data
	# if using not optimizing dummy vars, add random init condition for dummy dimensions
	if (S.opt_dummy_u0 == false) u0 = vcat(u0,randn(Float64,S.n-2)) end

	return ode_data, u0, tspan, tsteps, ode_data_orig
end

# define diff eq functions

function setup_diffeq_func(S)
	# activation function to create nonlinearity, identity is no change
	activate = if (S.activate == 1) identity elseif (S.activate == 2) tanh
						elseif (S.activate == 3) sigmoid else swish end
	# If optimizing initial conditions for dummy dimensions, then for initial condition u0,
	# dummy dimensions are first entries of p
	predict_node_dummy(p, prob, u_init) =
	  		Array(prob(vcat(u_init,p[1:S.n-2]), p[S.n-1:end]))
	predict_node_nodummy(p, prob, u_init) = Array(prob(u_init, p))
	predict_ode_dummy(p, prob, u_init) =
			solve(prob, S.solver, u0=vcat(u_init,p[1:S.n-2]), p=p[S.n-1:end])
	predict_ode_nodummy(p, prob, u_init) = solve(prob, S.solver, p=p)

	# For NODE, many simple options to build alternative network architecture, see SciML docs
	if S.use_node
		dudt = FastChain(FastDense(S.n, S.layer_size, activate), FastDense(S.layer_size, S.n))
		ode! = nothing
		predict = S.opt_dummy_u0 ?
			predict_node_dummy :
			predict_node_nodummy
	else
		dudt = nothing
		function ode!(du, u, p, t, n, nsqr)
			s = reshape(p[1:nsqr], n, n)
			du .= activate.(s*u .- p[nsqr+1:end])
		end
		predict = S.opt_dummy_u0 ?
			predict_ode_dummy :
			predict_ode_nodummy
	end
	return dudt, ode!, predict
end

# This gets called for each iterate of DiffEqFlux.sciml_train(), args to left of ;
# are p plus the return values of loss(). Use this function for any intermediate
# plotting, display of status, or collection of statistics
# For more general plotting function, see FitODE_plots, plot_target_pred()
function callback(p, loss_val, S, L, pred; doplot = true, show_lines = false)
	# printing gradient takes calculation time, turn off may yield speedup
	if (S.print_grad)
		grad = gradient(p->loss(p,S,L)[1], p)[1]
		gnorm = sqrt(sum(abs2, grad))
		println(@sprintf("%5.3e; %5.3e", loss_val, gnorm))
	else
		display(loss_val)
	end
	if doplot
		# plot current prediction against data
		len = length(pred[1,:])
		ts = L.tsteps[1:len]
		plt = plot(size=(600,800), layout=(2,1))
		plot_type! = if show_lines plot! else scatter! end
		plot_type!(ts, L.ode_data[1,1:len], label = "hare", subplot=1)
		plot_type!(plt, ts, pred[1,:], label = "pred", subplot=1)
		plot_type!(ts, L.ode_data[2,1:len], label = "lynx", subplot=2)
		plot_type!(plt, ts, pred[2,:], label = "pred", subplot=2)
		display(plot(plt))
  	end
  	return false
end

calc_gradient(p,S,L) = gradient(p->loss(p,S,L)[1], p)[1]

function loss(p, S, L)
	pred_all = L.predict(p, L.prob, L.u0)
	pred = pred_all[1:2,:]	# First rows are hare & lynx, others dummies
	pred_length = length(pred[1,:])
	if pred_length != length(L.w[1,:]) println("Mismatch") end
	loss = sum(abs2, L.w[:,1:pred_length] .* (L.ode_data[:,1:pred_length] .- pred))
	return loss, S, L, pred_all
end

# For iterative fitting of times series
function weights(a, tsteps; b=10.0, trunc=S.wt_trunc) 
	w = [1 - cdf(Beta(a,b),x/tsteps[end]) for x = tsteps]
	v = w[w .> trunc]'
	vcat(v,v)
end

function fit_diffeq(S)
	ode_data, u0, tspan, tsteps, ode_data_orig = FitODE.read_data(S);
	dudt, ode!, predict = FitODE.setup_diffeq_func(S);
	
	# If using subset of data for training then keep original and truncate tsteps
	tsteps_all = copy(tsteps)
	tspan_all = tspan
	if (S.train_frac < 1)
		tsteps = tsteps[tsteps .<= S.train_frac*tsteps[end]]
		tspan = (tsteps[begin], tsteps[end])
	end
	
	beta_a = 1:S.wt_incr:S.wt_steps
	if !S.use_node p_init = 0.1*rand(S.nsqr + S.n) end; # n^2 matrix plus n for individual growth
	
	local result
	for i in 1:length(beta_a)
		println("Iterate ", i, " of ", length(beta_a))
		w = weights(S.wt_base^beta_a[i], tsteps; trunc=S.wt_trunc)
		last_time = tsteps[length(w[1,:])]
		ts = tsteps[tsteps .<= last_time]
		# for ODE and opt_dummy, may redefine u0 and p, here just need right sizes for ode!
		prob = S.use_node ?
					NeuralODE(dudt, (0.0,last_time), S.solver, saveat = ts, 
						reltol = S.rtol, abstol = S.atol) :
					ODEProblem((du, u, p, t) -> ode!(du, u, p, t, S.n, S.nsqr), u0,
						(0.0,last_time), p_init, saveat = ts, reltol = S.rtol, abstol = S.atol)
		L = loss_args(u0,prob,predict,ode_data,ode_data_orig,tsteps,w)
		# On first time through loop, set up params p for optimization. Following loop
		# turns use the parameters returned from sciml_train(), which are in result.u
		if (i == 1)
			p = S.use_node ? prob.p : p_init
			if S.opt_dummy_u0 p = vcat(randn(S.n-2),p) end
		else
			p = result.u
		end
		result = DiffEqFlux.sciml_train(p -> loss(p,S,L),
						 p, ADAM(S.adm_learn); cb = callback, maxiters=S.max_it)
	end
	# To prepare for final fitting and calculations, must set prob to full training
	# period with tspan and tsteps and then redefine loss_args values in L
	prob = S.use_node ?
			NeuralODE(dudt, tspan, S.solver, saveat = tsteps, 
					reltol = S.rtolR, abstol = S.atolR) :
			ODEProblem((du, u, p, t) -> ode!(du, u, p, t, S.n, S.nsqr), u0,
					tspan, p_init, saveat = tsteps, reltol = S.rtolR, abstol = S.atolR)
	if (S.train_frac == 1.0)
		prob_all = prob
	else
		prob_all = S.use_node ?
			NeuralODE(dudt, tspan_all, S.solver, saveat = tsteps_all, 
					reltol = S.rtolR, abstol = S.atolR) :
			ODEProblem((du, u, p, t) -> ode!(du, u, p, t, S.n, S.nsqr), u0, tspan_all, 
					p_init, saveat = tsteps_all, reltol = S.rtolR, abstol = S.atolR)
		
	end
	w = ones(2,length(tsteps))
	L = loss_args(u0,prob,predict,ode_data,ode_data_orig,tsteps,w)
	A = all_time(prob_all, tsteps_all)
	p_opt = refine_fit(result.u, S, L)
	return p_opt, L, A
end

function refine_fit(p, S, L; rate_div=5.0, iter_mult=2.0)
	println("\nFinal round of fitting, using full time series in given data")
	println("Last step of previous fit did not fully weight final pts in series")
	println("Reducing ADAM learning rate by ", rate_div,
				" and increasing iterates by ", iter_mult, "\n")
	if S.use_node
		println("Also reducing tolerances to rel = ", S.rtolR, " and abs = ", S.atolR)
		println("which if set too low may cause a delay for NODE\n")
		println("If delay is too long, adjust S.rtolR and S.atolR as needed\n")
	end
	rate = S.adm_learn / rate_div
	iter = S.max_it * iter_mult
	result = DiffEqFlux.sciml_train(p -> loss(p,S,L),
						 p, ADAM(rate); cb = callback, maxiters=iter)
	return result.u
end

function refine_fit_bfgs(p, S, L) 
	println("\nBFGS sometimes suffers instability or gives other warnings")
	println("If so, then abort and do not use result\n")
	result = DiffEqFlux.sciml_train(p -> loss(p,S,L),
						 p, BFGS(); cb = callback, maxiters=S.max_it)
	return result.u
end

# particular data saved changes with versions of program
# could send tuple rather than naming variables but helps to have
# named variables here to check that currently required variables are
# saved
save_data(p, S, L, L_all, loss_v, pred; file=S.out_file) =
			jldsave(file; p, S, L, L_all, loss_v, pred)

# jld2 stores data as Dict with keys as strings
# to return named tuple, must first transform to dict with keys as symbols
#	see https://www.reddit.com/r/Julia/comments/8aw93w
# then use ... operator to transform dict w/symbol keys to named tuple
#	see https://discourse.julialang.org/t/10899/8?page=2
# Disadvantage of loading whatever is saved is that no checking for particular
# variables is made, so caller must check that required variables present
function load_data(file)
	global load_data_warning
	dt_string_keys = load(file)
	dt_symbol_keys = Dict()
	for (k,v) in dt_string_keys
    	dt_symbol_keys[Symbol(k)] = v
	end
	if load_data_warning
		println("\nWarning may occur if loaded data struct not defined or")
		println("differs from current definition of that structure. Check keys")
		println("in returned named tuple by using keys() on returned value. Check")
		println("if a required key is missing and adjust accordingly.\n")
		load_data_warning = false
	end
	(; dt_symbol_keys...)
end

# Can check for specific keys in load by this function, if required key
# is missing, should throw an error
function load_data_old(file)
	dt = load(file)
	(p = dt["p"], S = dt["S"], L = dt["L"], L_all = dt["L_all"],
						loss_v = dt["loss_v"], pred = dt["pred"])
end

end # module