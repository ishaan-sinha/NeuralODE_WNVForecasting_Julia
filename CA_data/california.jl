module California

using DataFrames, CSV, Dates, Statistics, Plots

features = [:count]
units = ["people"]
feature_names = ["Count"]

function load()
    df = CSV.read(pwd() * "/CA_data/wnv_data.csv", DataFrame)
    df = df[:, [:Column1, :count]]
    df = rename(df, :Column1 => :date)
end

df = load()


"""Plots each feature as a time series."""
function plot_features(df)
    plots = map(enumerate(zip(features, feature_names, units))) do (i, (f, n, u))
        plot(df[:, :date], df[:, f],
             title=n, label=nothing,
             ylabel=u, size=(800, 600),
             color=i)
    end

    n = length(plots)
    plot(plots..., layout=(floor(Int,n / 2), floor(Int,n / 2)))
end

plot_features(df)


function normalize(x)
    μ = mean(x)
    σ = std(x)
    z = (x .- μ) ./ σ
    return z, μ, σ
end


function preprocess(raw_df, num_train=100)
    raw_df[:,:year] = Float64.(year.(raw_df[:,:date]))
    raw_df[:,:month] = Float64.(month.(raw_df[:,:date]))

    df = combine(
        groupby(raw_df, [:year, :month]),
        :date => (d -> mean(year.(d)) .+ mean(month.(d)) ./ 12),
        :count => :count,
        renamecols=false
    )
    
    t_and_y(df) = df[!, :date], Matrix(df[!, features])
    t_train, y_train = t_and_y(df[1:num_train,:])
    t_test, y_test = t_and_y(df[num_train+1:end,:])
    print(t_train)
    
    t_train, t_mean, t_scale = normalize(t_train)
    y_train, y_mean, y_scale = normalize(y_train)
   
    t_test = (t_test .- t_mean) ./ t_scale
    y_test = (y_test .- y_mean) ./ y_scale

    return (
        vec(t_train), y_train,
        vec(t_test),  y_test,
        (t_mean, t_scale),
        (y_mean, y_scale)
    )
    
end


end