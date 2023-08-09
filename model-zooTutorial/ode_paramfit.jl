using DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqSensitivity, Plots

function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob_ode = ODEProblem(lotka_volterra!, u0, tspan, p)
sol_ode = solve(prob_ode, Tsit5())

# Plot the solution
using Plots
plot(sol_ode)

# Create a solution (prediction) for a given starting point u0 and set of
# parameters p
function predict_adjoint(p)
    return Array(concrete_solve(prob_ode, Tsit5(), u0, p, saveat = tsteps))
  end

function loss_adjoint(p)
    prediction = predict_adjoint(p)
    loss = sum(abs2, x-(x^2-2) for x in prediction)
    return loss, prediction
end

# Callback function to observe training
list_plots = []
iter = 0
callback = function (p, l, pred)
  global iter

  if iter == 0
    list_plots = []
  end
  iter += 1

  display(l)

  # using `remake` to re-create our `prob` with current parameters `p`
  remade_solution = solve(remake(prob_ode, p = p), Tsit5(), saveat = tsteps)
  plt = plot(remade_solution, ylim = (0, 6))

  push!(list_plots, plt)
  display(plt)

  # Tell sciml_train to not halt the optimization. If return true, then
  # optimization stops.
  return false
end

result_ode = DiffEqFlux.sciml_train(loss_adjoint, p,
                                    BFGS(initial_stepnorm = 0.0001),
                                    cb = callback)

