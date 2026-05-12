# # Double Gyre

using Oceananigans
using Oceananigans.Units
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: FunctionField
using Oceananigans.Grids: φnode, nodes, xspacings, yspacings, zspacings
using Oceananigans.Architectures: on_architecture
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, ImmersedBoundaryCondition, PartialCellBottom

using CairoMakie
using NCDatasets
using Statistics
using Printf

# Architecture: CPU() or GPU(); the latter requires using CUDA package
#using CUDA
arch = CPU()

const λ_west = -30 # [°] longitude of west boundary
const λ_east = +30 # [°] longitude of east boundary
const φ_south = 15 # [°] latitude of south boundary
const φ_north = 75 # [°] latitude of north boundary
const φ₀ = (φ_south + φ_north) / 2 # [°] latitude of the center of the domain

const Lλ = λ_east - λ_west   # [°] longitude extent of the domain
const Lφ = φ_north - φ_south # [°] latitude extent of the domain
const Lz = 2kilometers # depth [m]

# timestep and final time
Δt = 30minutes # adjust depending on chosen resolution; 30min seems OK with 1/4 deg resolution + RK3 timestep
stop_time = 20 * 365days

# resolution
resolution = 4 # corresponds to 1/resolution in degrees
Nλ = Integer(Lλ * resolution)
Nφ = Integer(Lφ * resolution)
Nz = 35
use_sloping_sidewalls = true

# bottom drag type: :linear or :quadratic
drag_type = :linear

underlying_grid = LatitudeLongitudeGrid(arch;
                                        size = (Nλ, Nφ, Nz),
                                        longitude = (λ_west, λ_east),
                                        latitude = (φ_south, φ_north),
                                        z = ExponentialDiscretization(Nz, -Lz, 0, scale=Lz/3),
                                        topology = (Bounded, Bounded, Bounded),
                                        halo = (7, 7, 4))

@inline tanh_ramp(ξ, sharpness) = (tanh(sharpness * (ξ - 0.5)) + tanh(sharpness / 2)) / (2 * tanh(sharpness / 2))

g  = Oceananigans.defaults.gravitational_acceleration
α  = 2e-4 # [K⁻¹] thermal expansion coefficient
cᵖ = 3991 # [J K⁻¹ kg⁻¹] heat capacity for seawater
ρ₀ = 1028 # [kg m⁻³] reference seawater density

Δzₛ = minimum_zspacing(underlying_grid) # vertical spacing at the surface [m]

parameters = (Lφ = Lφ,
              Lz = Lz,
              φ₀ = φ₀,           # latitude of the center of the domain [°]
               τ = 0.1 / ρ₀,     # surface kinematic wind stress [m² s⁻²]
               μ = 0.001,        # bottom drag damping parameter [m s⁻¹]
      drag_type = drag_type,    # :linear or :quadratic
     λ_slope_width = 7.5,        # west/east sidewall width [°]
     φ_slope_width = 7.5,        # south/north sidewall width [°]
    slope_sharpness = 3.5,       # nondimensional steepness of tanh sidewall transition
              Δb = 30 * α * g,   # surface vertical buoyancy gradient [s⁻²]
       timescale = 30days,       # relaxation time scale [s]
              vˢ = Δzₛ / 30days) # buoyancy pumping velocity [m s⁻¹]

function sidewall_bathymetry(λ, φ, p)
    ξx = clamp(min((λ - λ_west) / p.λ_slope_width,
                   (λ_east - λ) / p.λ_slope_width), 0, 1)
    ξy = clamp(min((φ - φ_south) / p.φ_slope_width,
                   (φ_north - φ) / p.φ_slope_width), 0, 1)

    # Euclidean blending removes the crease at corners by combining the
    # distances to the adjacent walls into a single smooth radial coordinate.
    ξ = clamp(sqrt(ξx^2 + ξy^2), 0, 1)
    return -p.Lz * tanh_ramp(ξ, p.slope_sharpness)
end

if use_sloping_sidewalls
    immersed_boundary = PartialCellBottom((λ, φ) -> sidewall_bathymetry(λ, φ, parameters);
                                          minimum_fractional_cell_height = 0.2)
    grid = ImmersedBoundaryGrid(underlying_grid, immersed_boundary)
else
    grid = underlying_grid
end

# ## Boundary conditions
#
# ### Wind stress
@inline u_stress(λ, φ, t, p) = p.τ * sin(2π * (φ - p.φ₀) / p.Lφ)

# ### Buoyancy relaxation
@inline surface_buoyancy(φ, p) = p.Δb * (φ - p.φ₀) / p.Lφ
@inline function buoyancy_relaxation(i, j, grid, clock, model_fields, p)
    b = @inbounds model_fields.b[i, j, grid.Nz] # surface b
    φ = φnode(j, grid, Center())
    return - 1 / p.timescale * (b - surface_buoyancy(φ, p))
end

@inline function u_drag(i, j, grid, clock, model_fields, p)
    u = @inbounds model_fields.u[i, j, 1]
    if p.drag_type == :quadratic
        return - p.μ * abs(u) * u
    else  # linear
        return - p.μ * u
    end
end

@inline function v_drag(i, j, grid, clock, model_fields, p)
    v = @inbounds model_fields.v[i, j, 1]
    if p.drag_type == :quadratic
        return - p.μ * abs(v) * v
    else  # linear
        return - p.μ * v
    end
end

@inline function u_immersed_drag(i, j, k, grid, clock, model_fields, p)
    u = @inbounds model_fields.u[i, j, k]
    if p.drag_type == :quadratic
        return - p.μ * abs(u) * u
    else  # linear
        return - p.μ * u
    end
end

@inline function v_immersed_drag(i, j, k, grid, clock, model_fields, p)
    v = @inbounds model_fields.v[i, j, k]
    if p.drag_type == :quadratic
        return - p.μ * abs(v) * v
    else  # linear
        return - p.μ * v
    end
end

@inline function u_bottom_drag(i, j, k, grid, u, p)
    u★ = @inbounds u[i, j, 1]
    if p.drag_type == :quadratic
        return - p.μ * abs(u★) * u★
    else
        return - p.μ * u★
    end
end

@inline function v_bottom_drag(i, j, k, grid, v, p)
    v★ = @inbounds v[i, j, 1]
    if p.drag_type == :quadratic
        return - p.μ * abs(v★) * v★
    else
        return - p.μ * v★
    end
end

@inline function surface_buoyancy_forcing(i, j, k, grid, b, p)
    b★ = @inbounds b[i, j, grid.Nz]
    φ = φnode(j, grid, Center())
    return - 1 / p.timescale * (b★ - surface_buoyancy(φ, p))
end

u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=parameters)
v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=parameters)
u_immersed_drag_bc = FluxBoundaryCondition(u_immersed_drag, discrete_form=true, parameters=parameters)
v_immersed_drag_bc = FluxBoundaryCondition(v_immersed_drag, discrete_form=true, parameters=parameters)

u_stress_bc = FluxBoundaryCondition(u_stress; parameters)
b_relax_bc  = FluxBoundaryCondition(buoyancy_relaxation, discrete_form=true, parameters=parameters)

if use_sloping_sidewalls
    u_immersed_bcs = ImmersedBoundaryCondition(bottom = u_immersed_drag_bc)
    v_immersed_bcs = ImmersedBoundaryCondition(bottom = v_immersed_drag_bc)

    u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc, immersed = u_immersed_bcs)
    v_bcs = FieldBoundaryConditions(                   bottom = v_drag_bc, immersed = v_immersed_bcs)
else
    u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
    v_bcs = FieldBoundaryConditions(                   bottom = v_drag_bc)
end
b_bcs = FieldBoundaryConditions(top = b_relax_bc)

# ## Turbulence closure
boundary_layer_closure     = RiBasedVerticalDiffusivity()
vertical_diffusive_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ = 3e-5, ν = 5e-4)

closures = (boundary_layer_closure, vertical_diffusive_closure)

# ## Model building
model = HydrostaticFreeSurfaceModel(; grid,
                                    free_surface = SplitExplicitFreeSurface(grid; cfl = 0.7),
                                    timestepper = :SplitRungeKutta3,
                                    momentum_advection = WENOVectorInvariant(),
                                    tracer_advection = WENO(),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    closure  = closures,
                                    tracers  = :b, # if boundary_layer_closure = RiBasedVerticalDiffusivity()
                                    # tracers  = (:b, :e), # if boundary_layer_closure = CATKEVerticalDiffusivity()
                                    boundary_conditions = (u = u_bcs, v = v_bcs, b = b_bcs))

# ## Initial conditions
bᵢ(λ, φ, z) = parameters.Δb * z / grid.Lz
set!(model, b = bᵢ)

# ## Simulation setup
simulation = Simulation(model; Δt, stop_time)

# add progress callback
wall_clock = [time_ns()]

function progress(sim)
    @info @sprintf("[%05.2f%%] i: %d, t: %s, max(u): (%6.2e, %6.2e, %6.2e) m s⁻¹, Δt: %s, wall time: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt),
            prettytime(1e-9 * (time_ns() - wall_clock[1])))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, TimeInterval(7days))

# ## Output

metric_grid(grid) = grid
metric_grid(grid::ImmersedBoundaryGrid) = grid.underlying_grid

function horizontal_metric_array(operation)
    field = compute!(Field(operation))
    return Array(interior(field, :, :, 1))
end

function vertical_metric_array(operation)
    field = compute!(Field(operation))
    return vec(Array(interior(field, 1, 1, :)))
end

function define_coordinate!(ds, name, values; long_name, units)
    defDim(ds, name, length(values))
    variable = defVar(ds, name, Float64, (name,),
                      attrib = Dict("long_name" => long_name, "units" => units))
    variable[:] = Array(values)
    return nothing
end

function define_static_variable!(ds, name, values, dims; long_name, units)
    variable = defVar(ds, name, Float64, dims,
                      attrib = Dict("long_name" => long_name, "units" => units))
    variable[ntuple(_ -> Colon(), ndims(values))...] = values
    return nothing
end

function write_static_grid_file(grid; filename = "double_gyre_grid.nc")
    grid = metric_grid(grid)

    λc, φc, zc = nodes(grid, Center(), Center(), Center())
    λf, φf, zf = nodes(grid, Face(),   Face(),   Face())

    NCDataset(filename, "c") do ds
        ds.attrib["title"] = "Static grid metrics for double gyre simulation"
        ds.attrib["description"] = "Horizontal spacings are saved at all tracer and velocity C-grid horizontal locations."

        define_coordinate!(ds, "lon_c", λc; long_name = "Longitude at cell centers", units = "degrees_east")
        define_coordinate!(ds, "lon_f", λf; long_name = "Longitude at cell faces",   units = "degrees_east")
        define_coordinate!(ds, "lat_c", φc; long_name = "Latitude at cell centers",  units = "degrees_north")
        define_coordinate!(ds, "lat_f", φf; long_name = "Latitude at cell faces",    units = "degrees_north")
        define_coordinate!(ds, "z_c",   zc; long_name = "Height at cell centers",    units = "m")
        define_coordinate!(ds, "z_f",   zf; long_name = "Height at cell faces",      units = "m")

        define_static_variable!(ds, "dx_cc", horizontal_metric_array(xspacings(grid, Center(), Center())),
                                ("lon_c", "lat_c"); long_name = "Zonal spacing at tracer points", units = "m")
        define_static_variable!(ds, "dx_fc", horizontal_metric_array(xspacings(grid, Face(), Center())),
                                ("lon_f", "lat_c"); long_name = "Zonal spacing at u points", units = "m")
        define_static_variable!(ds, "dx_cf", horizontal_metric_array(xspacings(grid, Center(), Face())),
                                ("lon_c", "lat_f"); long_name = "Zonal spacing at v points", units = "m")
        define_static_variable!(ds, "dx_ff", horizontal_metric_array(xspacings(grid, Face(), Face())),
                                ("lon_f", "lat_f"); long_name = "Zonal spacing at horizontal cell corners", units = "m")

        define_static_variable!(ds, "dy_cc", horizontal_metric_array(yspacings(grid, Center(), Center())),
                                ("lon_c", "lat_c"); long_name = "Meridional spacing at tracer points", units = "m")
        define_static_variable!(ds, "dy_fc", horizontal_metric_array(yspacings(grid, Face(), Center())),
                                ("lon_f", "lat_c"); long_name = "Meridional spacing at u points", units = "m")
        define_static_variable!(ds, "dy_cf", horizontal_metric_array(yspacings(grid, Center(), Face())),
                                ("lon_c", "lat_f"); long_name = "Meridional spacing at v points", units = "m")
        define_static_variable!(ds, "dy_ff", horizontal_metric_array(yspacings(grid, Face(), Face())),
                                ("lon_f", "lat_f"); long_name = "Meridional spacing at horizontal cell corners", units = "m")

        define_static_variable!(ds, "dz_c", vertical_metric_array(zspacings(grid, Center())),
                                ("z_c",); long_name = "Vertical spacing at cell centers", units = "m")
        define_static_variable!(ds, "dz_f", vertical_metric_array(zspacings(grid, Face())),
                                ("z_f",); long_name = "Vertical spacing at cell faces", units = "m")
    end

    return nothing
end

u, v, w = model.velocities
b = model.tracers.b

speed = Field(u^2 + v^2)
buoyancy_variance = Field(b^2)

outputs = merge(model.velocities, model.tracers, (speed = speed, b² = buoyancy_variance))

simulation.output_writers[:fields] = JLD2Writer(model, outputs,
                                                schedule = TimeInterval(7days),
                                                filename = "double_gyre",
                                                indices = (:, :, model.grid.Nz),
                                                overwrite_existing = true)

barotropic_u = Field(Average(model.velocities.u, dims = 3))
barotropic_v = Field(Average(model.velocities.v, dims = 3))

simulation.output_writers[:barotropic_velocities] =
    JLD2Writer(model, (u = barotropic_u, v = barotropic_v),
               schedule = AveragedTimeInterval(30days, window = 10days),
               filename = "double_gyre_circulation",
               overwrite_existing = true)

tau_s = FunctionField{Face, Center, Nothing}(u_stress, grid; clock = model.clock, parameters)
tau_b_u = Field(KernelFunctionOperation{Face, Center, Nothing}(u_bottom_drag, grid, u, parameters))
tau_b_v = Field(KernelFunctionOperation{Center, Face, Nothing}(v_bottom_drag, grid, v, parameters))
surface_buoyancy_flux = Field(KernelFunctionOperation{Center, Center, Nothing}(surface_buoyancy_forcing, grid, b, parameters))

monthly_outputs = (u = u,
                   v = v,
                   w = w,
                   b = b,
                   tau_s = tau_s,
                   tau_b_u = tau_b_u,
                   tau_b_v = tau_b_v,
                   surface_buoyancy_forcing = surface_buoyancy_flux)

monthly_output_attributes = Dict(
    "tau_s" => Dict("long_name" => "Surface zonal kinematic wind stress", "units" => "m2 s-2"),
    "tau_b_u" => Dict("long_name" => "Bottom zonal kinematic drag", "units" => "m2 s-2"),
    "tau_b_v" => Dict("long_name" => "Bottom meridional kinematic drag", "units" => "m2 s-2"),
    "surface_buoyancy_forcing" => Dict("long_name" => "Surface buoyancy forcing", "units" => "m s-3"))

simulation.output_writers[:monthly_means] =
    NetCDFWriter(model, monthly_outputs;
                 schedule = AveragedTimeInterval(30days),
                 filename = "double_gyre_monthly_means.nc",
                 include_grid_metrics = false,
                 output_attributes = monthly_output_attributes,
                 overwrite_existing = true)

write_static_grid_file(grid)

run!(simulation)


# # A neat movie

# We open the JLD2 file, and extract the `grid` and the iterations we ended up saving at.

filename = "double_gyre.jld2"
u_timeseries = FieldTimeSeries(filename, "u"; architecture = CPU())
v_timeseries = FieldTimeSeries(filename, "v"; architecture = CPU())
s_timeseries = FieldTimeSeries(filename, "speed"; architecture = CPU())

times = u_timeseries.times

λᵤ, φᵤ, zᵤ = nodes(u_timeseries[1])
λᵥ, φᵥ, zᵥ = nodes(v_timeseries[1])
λₛ, φₛ, zₛ = nodes(s_timeseries[1])

# Finally, we're ready to animate.

@info "Making an animation from the saved data..."

n = Observable(1)

u = @lift interior(u_timeseries[$n], :, :)
v = @lift interior(v_timeseries[$n], :, :)
s = @lift interior(s_timeseries[$n], :, :)

extrema_reduction_factor = 0.5

ulims = extrema(u_timeseries.data) .* extrema_reduction_factor
vlims = extrema(v_timeseries.data) .* extrema_reduction_factor
slims = extrema(s_timeseries.data) .* extrema_reduction_factor

fig = Figure(size = (1650, 1250))

title_u = @lift "Zonal Velocity after " *string(round(times[$n]/day, digits = 1))*" days"
ax_u = Axis(fig[1:2, 1]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)")
hm_u = heatmap!(ax_u, λᵤ, φᵤ, u; colorrange = ulims, colormap = :balance)
Colorbar(fig[1:2, 2], hm_u; label = "Zonal velocity (m s⁻¹)")

title_v = @lift "Meridional Velocity after " *string(round(times[$n]/day, digits = 1))*" days"
ax_v = Axis(fig[3:4, 1]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)")
hm_v = heatmap!(ax_v, λᵥ, φᵥ, v; colorrange = vlims, colormap = :balance)
Colorbar(fig[3:4, 2], hm_v; label = "Meridional velocity (m s⁻¹)")

title_s = @lift "Speed after " *string(round(times[$n]/day, digits = 1))*" days"
ax_s = Axis(fig[2:3, 3]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)")
hm_s = heatmap!(ax_s, λₛ, φₛ, s; colorrange = slims, colormap = :balance)
Colorbar(fig[2:3, 4], hm_s; label = "Speed (m s⁻¹)")

frames = 1:length(times)

CairoMakie.record(fig, filename[1:end-5] * ".mp4", frames, framerate = 8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end


# Plot the barotropic circulation

filename_barotropic = "double_gyre_circulation.jld2"

U_timeseries = FieldTimeSeries(filename_barotropic, "u"; grid, architecture = CPU())
V_timeseries = FieldTimeSeries(filename_barotropic, "v"; grid, architecture = CPU())

# time-average; adjust accordingly to avoid spinup
U_mean = Field{Oceananigans.Fields.location(U_timeseries)...}(on_architecture(CPU(), grid))
V_mean = Field{Oceananigans.Fields.location(V_timeseries)...}(on_architecture(CPU(), grid))

for (iter, time_snapshop) in enumerate(round(Int, length(U_timeseries)/2):length(U_timeseries))
    parent(U_mean) .= parent(U_mean) * (iter - 1) / iter .+ parent(U_timeseries[time_snapshop]) / iter
    parent(V_mean) .= parent(V_mean) * (iter - 1) / iter .+ parent(V_timeseries[time_snapshop]) / iter
end

fig = Figure(size = (1650, 1250))

title_U = "Depth- and Time-Averaged Zonal Velocity"
ax_U = Axis(fig[1:2, 1]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)")
hm_U = heatmap!(ax_U, λᵤ, φᵤ, U_mean; colorrange = ulims ./10, colormap = :balance)
Colorbar(fig[1:2, 2], hm_U)

title_V = "Depth- and Time-Averaged Meridional Velocity"
ax_V = Axis(fig[3:4, 1]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)")
hm_V = heatmap!(ax_V, λᵥ, φᵥ, V_mean; colorrange = vlims ./10, colormap = :balance)
Colorbar(fig[3:4, 2], hm_V)

Ψ = CumulativeIntegral(- U_mean, dims = 2) |> Field
Ψlims = extrema(Ψ) .* extrema_reduction_factor

title_Ψ = "Barotropic Streamfunction"
ax_Ψ = Axis(fig[2:3, 3]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)")
hm_Ψ = heatmap!(ax_Ψ, λᵤ, φᵤ, Ψ; colorrange = Ψlims, colormap = :balance)
Colorbar(fig[2:3, 4], hm_Ψ)

save("double_gyre_circulation.png", fig)
