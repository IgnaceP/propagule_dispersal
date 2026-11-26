#=
read_slf:
- Julia version:
- Author: ignace
- Date: 2026-06-11
=#

using GLMakie
using Random
using ProgressBars
using JLD2
using NCDatasets
using CSV
using DataFrames
using Interpolations



include("tin_interpolation.jl")


fn = "/Volumes/Extreme SSD/Laura/run03_surface_velocity.nc"
wind_fn = "/Volumes/Extreme SSD/Laura/Wind_Hansweert_2020.txt"

dt = 5*60
beta = 0.1

# -------------------------------------------------------------------------------------------------- #
# Load data and set attributes
# -------------------------------------------------------------------------------------------------- #

ds = NCDataset(fn)

# get ikle field
connectivity = permutedims(ds["ikle"][:,:]) .+ 1

# get node xy
xy = hcat(ds["xcoor"][:], ds["ycoor"][:])

# get number of nodes
npoin = ds.dim["npoin"]

# get total time
ttime = ds["time"][end]
println("Total time: $(ttime/3600) hours ")

# get time step
ncdt = ds["time"][2] -  ds["time"][1]

# Precompute vertex-to-triangle mapping
vertex_to_triangles = Dict{Int, Vector{Int}}()
for (i, triangle) in enumerate(eachrow(connectivity))
    for vertex in triangle
        if !haskey(vertex_to_triangles, vertex)
            vertex_to_triangles[vertex] = []
        end
        push!(vertex_to_triangles[vertex], i)
    end
end

steps = Int(floor(ttime // dt))

# -------------------------------------------------------------------------------------------------- #
# Load wind data
# -------------------------------------------------------------------------------------------------- #

# load txt file
wind = CSV.read(wind_fn, DataFrame; delim=' ', header = 2, skipto = 4)

# 1 september is the 245 day of the year (2020 was a leap year)
wind = wind[(wind.T .>= 244*24*3600) .& (wind.T .< (245 + 92)*24*3600),:]
wind.T .-= 244*24*3600

# interpolate to dt
intx = LinearInterpolation(wind.T, wind.WINDX)
inty = LinearInterpolation(wind.T, wind.WINDY)

# Generate new time steps with the desired dt
wind_time = 0:dt:maximum(wind.T)

# Interpolate wind data to the new time steps
new_wind_x = intx.(wind_time)
new_wind_y = inty.(wind_time)

# Combine the interpolated data into a new DataFrame
wind = DataFrame(T=wind_time, u=new_wind_x, v=new_wind_y)

# -------------------------------------------------------------------------------------------------- #
# Release locations
# -------------------------------------------------------------------------------------------------- #

# generate random release locations
#randi = randperm(npoin)[1:10]
randi = JLD2.load_object("1000_random_node_ids.jld2")
poi_x, poi_y = 7.0e4, 3.775e5
pois = xy[randi, :]


#s = fill(Float32(NaN), size(pois, 1), steps)
xs = fill(Float32(NaN), steps, steps)
#ys = fill(Float32(NaN), size(pois, 1), steps)
ys = fill(Float32(NaN), steps, steps)


#xs[:,1] = pois[:,1]
#ys[:,1] = pois[:,2]

for i in 1:steps
    xs[i,i] = poi_x
    ys[i,i] = poi_y
end

global floating = fill(false, size(xs, 1))

# -------------------------------------------------------------------------------------------------- #
# The loop locations
# -------------------------------------------------------------------------------------------------- #

# pre-allocate for efficient memory usage
xysi = zeros(Float32, size(xs, 1), 2)
velocities = zeros(Float32, size(ds["VELOCITY U"], 1), 2)

# !!! check whether NN faster is with nearest neighbor search or with barycentric coordinates

for i in ProgressBar(2:steps)

            # get the seconds since the start of the simulation (1 september 00:00)
            wind_t = dt*(i-1)

            # get the velocities at the current time step
            velocities[:, 1] .= ds["VELOCITY U"][:, i-1]
            velocities[:, 2] .= ds["VELOCITY V"][:, i-1]

            # get the particle locations at the previous time step
            xysi[:, 1] .= xs[:, i-1]
            xysi[:, 2] .= ys[:, i-1]

            # compute interpolated velocities at the particle locations
            vi = tin_interpolator(xysi,
            xy, connectivity, velocities,
            vertex_to_triangles)

            na_mask = isnan.(xs[:, i]) .|| isnan.(ys[:, i])

            # get wind vector
            uw = wind.u[i]
            vw = wind.v[i]

            # update positions
            xs[na_mask,i] = xs[na_mask,i-1] .+ vi[na_mask,1] * dt .+ uw * dt * beta
            ys[na_mask,i] = ys[na_mask,i-1] .+ vi[na_mask,2] * dt .+ vw * dt * beta

end

# -------------------------------------------------------------------------------------------------- #
# Release locations
# -------------------------------------------------------------------------------------------------- #

fig = Figure(figsize=(10, 7))
ax1 = Axis(fig[1, 1], xlabel="x [m]", ylabel="y [m]")
mesh!(ax1, xy, connectivity, color = :gray, transparency=true, alpha=0.2)

for i in 1:size(xs, 1)
    lines!(ax1, xs[i,:], ys[i,:], color = :black, alpha = .05)
end
scatter!(ax1, xs[:,begin], ys[:,begin], color = :green, markersize=25, alpha = 1.)
scatter!(ax1, xs[:,end], ys[:,end], color = :red, markersize=5, alpha = 1.)

xlims!(10000,100000)
ylims!(330000, 400000)

display(fig)
save("trajectories.png", fig)  # Save as PNG
