module State

using DataFrames, CSV, Dates, Statistics, Plots

features = [:count]
units = ["people"]
feature_names = ["Count"]

function create_date(row)
    return Date(row.year, row.month, 1)  # Assuming day is the 1st of each month
end

function load(state)
    df = CSV.read(pwd() * "/state_data/rawData.csv", DataFrame)
    df = filter(row -> row[:state] == state, df)

    df.date = create_date.(eachrow(df))
    df = df[:, [:date, :count]]
end

function listOfStates()
    df = CSV.read(pwd() * "/state_data/rawData.csv", DataFrame)
    return unique(df.state)
end

function getPopulation(state, year)
    df = CSV.read(pwd() * "/state_data/statePopulations.csv", DataFrame)
    df = filter(row -> row[:state] == state, df)
    yearToGet = min(year, 2021)
    yearToGet = max(yearToGet, 2009)
    df = filter(row -> row[:year] == yearToGet, df)
    return df[!, :population][1]
end


end