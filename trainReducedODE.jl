
import Dates   
using Optim
using DifferentialEquations
using StatsPlots
using DataFrames
using CSV

include(("state_data/state.jl"))


#UPDATED MONTHLY PARAMETERS

q = 0.33
c = 0.9
PI_H = 900
b_2 = .09*(1-c*q)*30
BETA_3 = .88
MU_H = 1/840
ALPHA = 2
DELTA = 1/30

DFresult = DataFrame(state= String[], year = Int[], error = Float64[])
#for state in State.listOfStates()
for state in ["CA"]
    for year in [2016]
        odedata = State.load(state)
        odedata = filter(:date => x -> Dates.year(x) == year, odedata)
        odedata = odedata[6:12, :]

        population = State.getPopulation(state, year)

        function wnv(du, u, p, t)

            V_0, V_1, V_3 = p[1], p[2], p[3]
            #Model parameters
            M_i = V_0*sin(V_1 + t/V_3) #Infected mosquitoes
            #Current state
            S, E, I = u

            N_H = S + E + I
            b_2 = 0.09*(1-c*q)

            #Evaluate differential equations

            du[1] = PI_H - (b_2*BETA_3*M_i*S)/(N_H) - MU_H*S #S, susceptible humans
            du[2] = (b_2*BETA_3*M_i*S)/(N_H) - MU_H*E - E/ALPHA #E, asymptomatically infected humans
            
            du[3] = E/ALPHA - MU_H*I - DELTA*I#I, symptomatically infected humans
            
            #print("\n", du[3] , " ", year)

            print("\n E/ALPHA", E*ALPHA, " ", year)
            return nothing
        end

        function diffEqError(x)
            #set parameters and initial conditions
            u0 = [population, 0, 0]
            p = [x[1], x[2], x[3]]

            #V_0*sin(V_1 + t/V_3)
            if(x[1]*sin(x[2] + 1/x[3]) < 0)
                return Inf
            end
            if(x[1]*sin(x[2] + 6/x[3]) < 0)
                return Inf
            end


            t_end = 6
            tspan = (0.0, t_end)
            


            prob = ODEProblem(wnv, u0, tspan, p)
            sol = solve(prob, Rodas5(), saveat=1, dt=1e-6)
            #sol_pred = [sol[3,1], sol[3, trunc(Int,1/6 * t_end)], sol[3, trunc(Int, 2/6*t_end)], sol[3, trunc(Int, 3/6*t_end)], sol[3, trunc(Int, 4/6*t_end)], sol[3, trunc(Int, 5/6*t_end)], sol[3, trunc(Int, t_end)]]
            sol_pred = sol[3,:]
            print("\n", year, "\n")
            print("\n Predicted", sol_pred, " ", "\n")
            print("\n Observed", odedata[!, :count], " ", "\n")
            return sum(abs.(sol_pred .- odedata[!, :count]))
        end

        x = [10000.0, 2, 6]

        result = optimize(diffEqError, x, NelderMead())

        p = [result.minimizer[1], result.minimizer[2], result.minimizer[3]]

        u0 = [population, 0, 0]
        tspan = (0.0, 6)
        prob = ODEProblem(wnv, u0, tspan, p)
        sol = solve(prob, Rodas5(), saveat=1, dt=1e-6)

        plot(sol[3, :], label="Predicted")
        plot!(odedata[!, :count], label="Observed")
        savefig("[PREDICT]CA"*string(year)*".png")

        error = sum(abs.(sol[3,:] .- odedata[!, :count]))

        push!(DFresult, [state, year, error])
        #CSV.write("results.csv", DFresult)
    end

end
