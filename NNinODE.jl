
using DifferentialEquations

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.5f0)

t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))

dudt = Chain(x -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)
ps = Flux.params(n_ode)

pred = n_ode(u0) # Get the prediction using the correct initial condition
scatter(t,ode_data[1,:],label="data")
scatter!(t,pred[1,:],label="prediction")

function predict_n_ode()
    n_ode(u0)
  end
  loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())

  data = Iterators.repeated((), 1000)
  opt = ADAM(0.1)
  cb = function () #callback function to observe training
    display(loss_n_ode())
    # plot current prediction against data
    cur_pred = predict_n_ode()
    pl = scatter(t,ode_data[1,:],label="data")
    scatter!(pl,t,cur_pred[1,:],label="prediction")
    display(plot(pl))
  end
  
  # Display the ODE with the initial parameter values.
  cb()
  
  Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

  