using DifferentialEquations

function lotka_volterra(du,u,p,t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α*x - β*x*y
    du[2] = dy = -δ*y + γ*x*y
    end

#=
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p)
sol = solve(prob,Tsit5(),saveat=0.1)
A = sol[1,:] # length 101 vector


plot(sol)
t = 0:0.1:10.0
scatter!(t,A)
=#

using Flux, DiffEqFlux

p = [2.2, 1.0, 2.0, 0.4] # Initial Parameter Vector
params = Flux.Params([p])

function predict_rd() # Our 1-layer "neural network"
    solve(prob,Tsit5(),p=p,saveat=0.1)[1,:] # override with new parameters
  end

loss_rd() = sum(abs2,x-1 for x in predict_rd()) # loss function

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function () #callback function to observe training
    display(loss_rd())
    # using `remake` to re-create our `prob` with current parameters `p`
    display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6)))
  end
cb()
Flux.train!(loss_rd, params, data, opt, cb = cb)

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax)

m = Chain(
Dense(28^2, 32, relu),
# this would require an ODE of 32 parameters
p -> solve(prob,Tsit5(),p=p,saveat=0.1)[1,:],
Dense(32, 10),
softmax)





