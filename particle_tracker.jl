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
using Base.Threads

# print number of threads
print("Threads: ", Threads.nthreads(), "\n")

include("tin_interpolation.jl")


fn = "/Volumes/Extreme SSD/Laura/run03_surface_velocity.nc"
wind_fn = "/Volumes/Extreme SSD/Laura/Wind_Hansweert_2020.txt"

dt = 5*60
beta = 0.01

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
ttime = ds["time"][end] - ds["time"][1]
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

if dt % ncdt != 0
    error("timestep of the netcdf file is not a multiple of the desired dispersal dt")
end
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

# -------------------------------------------------------------------------------------------------- #
# The loop locations
# -------------------------------------------------------------------------------------------------- #

# pre-allocate for efficient memory usage
xysi = zeros(Float32, size(xs, 1), 2)
velocities = zeros(Float32, size(ds["VELOCITY U"], 1), 2)

# !!! check whether NN faster is with nearest neighbor search or with barycentric coordinates

for t in ProgressBar(2:steps)

            # get the seconds since the start of the simulation (1 september 00:00)
            wind_t = dt*(t-1)

            # float to integer
            nct = floor(Int, (t-1)*dt/ncdt)

            # get the velocities at the current time step
            velocities[:, 1] .= ds["VELOCITY U"][:, nct]
            velocities[:, 2] .= ds["VELOCITY V"][:, nct]

            # get the particle locations at the previous time step
            xysi[:, 1] .= xs[:, t-1]
            xysi[:, 2] .= ys[:, t-1]

            # compute interpolated velocities at the particle locations
            vi = tin_interpolator(xysi,
            xy, connectivity, velocities,
            vertex_to_triangles)

            # only update Nans (numbers indicate newly released particles)
            na_mask = isnan.(xs[:, t]) .|| isnan.(ys[:, t])

            # get relative wind vector
            uw = wind.u[t] .- vi[na_mask,1]
            vw = wind.v[t] .- vi[na_mask,2]

            # update positions
            xs[na_mask,t] = xs[na_mask,t-1] .+ vi[na_mask,1] * dt .+ uw * dt * beta
            ys[na_mask,t] = ys[na_mask,t-1] .+ vi[na_mask,2] * dt .+ vw * dt * beta

end

# -------------------------------------------------------------------------------------------------- #
# Release locations
# -------------------------------------------------------------------------------------------------- #

fig = Figure(figsize=(25, 15))
ax1 = Axis(fig[1, 1], xlabel="x [m]", ylabel="y [m]")
mesh!(ax1, xy, connectivity, color = :gray, transparency=true, alpha=0.2)

for i in 1:size(xs, 1)
    x_path = xs[i, :]
    x_path = x_path[.!isnan.(x_path)]
    y_path = ys[i, :]
    y_path = y_path[.!isnan.(y_path)]
    lines!(ax1, x_path, y_path, color = :black, alpha = .05)
    scatter!(ax1, x_path[1], y_path[1], color = :green, markersize=5, alpha = .25)
    scatter!(ax1, x_path[end], y_path[end], color = :red, markersize=5, alpha = .25)
end


xlims!(10000,100000)
ylims!(330000, 400000)

display(fig)
save("trajectories.png", fig)  # Save as PNG



#=
# Compute wind direction and magnitude
wind_direction = atan.(-1*wind.v, -1*wind.u) # Angle in radians
wind_magnitude = sqrt.(wind.u.^2 .+ wind.v.^2)

# Bin wind directions into 16 compass directions (optional)
num_bins = 16
bin_edges = range(-π, π; length=num_bins+1)
bin_counts = zeros(Float64, num_bins)

for dir in wind_direction
    bin_idx = findfirst(x -> dir < x, bin_edges)
    if bin_idx !== nothing
        bin_counts[bin_idx - 1] += 1  # Subtract 1 because `bin_edges` is 1-based
    end
end


# Add wind rose to the figure
fig_wind = Figure(figsize=(3, 3))
polar_ax = PolarAxis(fig_wind[1,1])
barplot!(polar_ax, bin_edges[1:end-1] .+ diff(bin_edges)/2, bin_counts; color=:blue, alpha=0.7, linewidth=0)
#display(fig_wind)

=#