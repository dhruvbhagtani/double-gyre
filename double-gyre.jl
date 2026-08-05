# # Double Gyre

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: φnode, inactive_node
using Oceananigans.Architectures: on_architecture
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, ImmersedBoundaryCondition, PartialCellBottom
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, Δx, Δy

using CairoMakie
using Statistics
using Printf

# Architecture: CPU() or GPU(); the latter requires using CUDA package
using CUDA
arch = GPU()

const λ_west = -30 # [°] longitude of west boundary
const λ_east = +30 # [°] longitude of east boundary
const φ_south = 15 # [°] latitude of south boundary
const φ_north = 75 # [°] latitude of north boundary
const φ₀ = (φ_south + φ_north) / 2 # [°] latitude of the center of the domain

const Lλ = λ_east - λ_west   # [°] longitude extent of the domain
const Lφ = φ_north - φ_south # [°] latitude extent of the domain
const Lz = 2kilometers # depth [m]
const coastal_depth = 500meters # minimum water depth at every lateral boundary [m]

# timestep and final time
Δt = 30minutes # adjust depending on chosen resolution; 30min seems OK with 1/4 deg resolution + RK3 timestep
model_year = 365days
model_month = model_year / 12
stop_time = 20 * model_year

# resolution
resolution = 4 # corresponds to 1/resolution in degrees
Nλ = Integer(Lλ * resolution)
Nφ = Integer(Lφ * resolution)
Nz = 35

# Bathymetry configuration, selected at submission with
# SLOPE_CONFIG=all, zonal, west, or none. The default keeps slopes only at
# the western and eastern boundaries.
@enum SlopeConfig::UInt8 NoSlopes WesternSlope ZonalSlopes AllSlopes

function parse_slope_config(value)
    name = lowercase(strip(value))
    name in ("none", "flat")                           && return NoSlopes
    name in ("west", "western")                        && return WesternSlope
    name in ("zonal", "east-west", "eastwest", "two") && return ZonalSlopes
    name in ("all", "four")                            && return AllSlopes
    throw(ArgumentError("Invalid SLOPE_CONFIG=$(repr(value)); expected all, zonal, west, or none"))
end

slope_config = parse_slope_config(get(ENV, "SLOPE_CONFIG", "zonal"))
println(stderr, "Using slope configuration: ", slope_config)
flush(stderr)

# Bottom drag type, selected at submission with DRAG_TYPE=linear, quadratic, or none.
# The UInt8-backed enum is safe to pass as a parameter to GPU kernels.
@enum DragType::UInt8 NoDrag LinearDrag QuadraticDrag

function parse_drag_type(value)
    name = lowercase(strip(value))
    name == "none"      && return NoDrag
    name == "linear"    && return LinearDrag
    name == "quadratic" && return QuadraticDrag
    throw(ArgumentError("Invalid DRAG_TYPE=$(repr(value)); expected linear, quadratic, or none"))
end

drag_type = parse_drag_type(get(ENV, "DRAG_TYPE", "quadratic"))
linear_drag_coefficient = 2e-4    # [m s⁻¹]
quadratic_drag_coefficient = 2e-3 # dimensionless

println(stderr, "Using bottom drag: ", drag_type)
if drag_type == LinearDrag
    println(stderr, "Linear drag coefficient: ", linear_drag_coefficient, " m s⁻¹")
elseif drag_type == QuadraticDrag
    println(stderr, "Quadratic drag coefficient: ", quadratic_drag_coefficient)
else
    println(stderr, "Bottom drag is disabled")
end
flush(stderr)

underlying_grid = LatitudeLongitudeGrid(arch;
                                        size = (Nλ, Nφ, Nz),
                                        longitude = (λ_west, λ_east),
                                        latitude = (φ_south, φ_north),
                                        z = ExponentialDiscretization(Nz, -Lz, 0, scale=Lz/3),
                                        topology = (Bounded, Bounded, Bounded),
                                        halo = (7, 7, 4))

# We can plot vertical spacing versus depth to inspect the prescribed grid stretching.

#=
fig = Figure()
ax = Axis(fig[1, 1],
          xlabel = "Vertical spacing (m)",
          ylabel = "Depth (m)",
          title = "Variation of Vertical Spacing with Depth")
scatterlines!(ax, grid.z.Δᵃᵃᶠ[1:Nz+1], grid.z.cᵃᵃᶠ[1:Nz+1])

save("double_gyre_grid_spacing.pdf", fig)
=#

g  = Oceananigans.defaults.gravitational_acceleration
α  = 2e-4 # [K⁻¹] thermal expansion coefficient
cᵖ = 3991 # [J K⁻¹ kg⁻¹] heat capacity for seawater
ρ₀ = 1028 # [kg m⁻³] reference seawater density

Δzₛ = minimum_zspacing(underlying_grid) # vertical spacing at the surface [m]
buoyancy_restoring_timescale = 30days

parameters = (Lφ = Lφ,
              Lz = Lz,
              φ₀ = φ₀,           # latitude of the center of the domain [°]
               τ = 0.05 / ρ₀,    # surface kinematic wind stress [m² s⁻²]
        linear_drag = linear_drag_coefficient,
     quadratic_drag = quadratic_drag_coefficient,
      drag_type = drag_type,     # NoDrag, LinearDrag, or QuadraticDrag
     λ_slope_width = 7.5,        # west/east sidewall width [°]
     φ_slope_width = 7.5,        # south/north sidewall width [°]
   coastal_depth = coastal_depth,
      slope_config = slope_config,
              Δb = 30 * α * g,   # surface vertical buoyancy gradient [s⁻²]
       timescale = buoyancy_restoring_timescale,       # relaxation time scale [s]
              vˢ = Δzₛ / buoyancy_restoring_timescale) # buoyancy pumping velocity [m s⁻¹]

@inline quintic_smoothstep(ξ) = ξ^3 * (10 + ξ * (-15 + 6ξ))

@inline function wall_factor(distance, width)
    ξ = clamp(distance / width, 0, 1)
    return quintic_smoothstep(ξ)
end

@inline function sidewall_bathymetry(λ, φ, p)
    west  = wall_factor(λ - λ_west,  p.λ_slope_width)
    east  = wall_factor(λ_east - λ,  p.λ_slope_width)
    south = wall_factor(φ - φ_south, p.φ_slope_width)
    north = wall_factor(φ_north - φ, p.φ_slope_width)

    # Multiplication blends adjacent wall slopes without the diagonal corner
    # creases produced by a hard minimum. The quintic factors have zero first
    # and second derivatives where they meet both the coast and flat bottom.
    interior_factor = if p.slope_config == AllSlopes
        west * east * south * north
    elseif p.slope_config == ZonalSlopes
        west * east
    else # WesternSlope
        west
    end

    return -p.coastal_depth - (p.Lz - p.coastal_depth) * interior_factor
end

if slope_config != NoSlopes
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
    # Oceananigans applies a top flux Qᵇ as ∂ₜ b = -Qᵇ / Δz.
    return p.vˢ * (b - surface_buoyancy(φ, p))
end

# ### Plotting surface forcing functions
#=
φ = grid.φᵃᶜᵃ[1:grid.Ny]
fig = Figure()
ax  = Axis(fig[1, 1],
           xlabel = "Buoyancy Profile",
           ylabel = "Latitude (Degree)",
           title = "Surface Buoyancy Forcing")
scatterlines!(ax, surface_buoyancy.(φ, Ref(parameters)), φ)

save("SurfaceBuoyancyForcing.pdf", fig)


fig = Figure()
ax = Axis(fig[1, 1],
          xlabel = "Wind Stress Profile",
          ylabel = "Latitude (Degree)",
          title = "Surface Wind Stress")
scatterlines!(ax, u_stress.(0, φ, 0, Ref(parameters)), φ)

save("SurfaceWindStress.pdf", fig)
=#

# ### Bottom drag
# Linear drag: -r * u, where r has units of m s⁻¹
# Quadratic drag: -Cᴅ * |u| * u, where Cᴅ is dimensionless
# No drag: zero momentum flux
@inline function drag_flux(component, speed, p)
    if p.drag_type == QuadraticDrag
        return -p.quadratic_drag * speed * component
    elseif p.drag_type == LinearDrag
        return -p.linear_drag * component
    else
        return zero(component)
    end
end

@inline horizontal_speed(u, v) = sqrt(u^2 + v^2)

# Evaluate drag in discrete form so field access and interpolation remain
# statically resolvable when these functions are compiled for the GPU. Since
# u and v are staggered, interpolate the cross-component to the native location
# of the component whose bottom flux is being computed.
@inline function u_drag(i, j, grid, clock, model_fields, p)
    k = 1
    uᵢ = @inbounds model_fields.u[i, j, k]
    vᵢ = ℑxyᶠᶜᵃ(i, j, k, grid, model_fields.v)
    return drag_flux(uᵢ, horizontal_speed(uᵢ, vᵢ), p)
end

@inline function v_drag(i, j, grid, clock, model_fields, p)
    k = 1
    uᵢ = ℑxyᶜᶠᵃ(i, j, k, grid, model_fields.u)
    vᵢ = @inbounds model_fields.v[i, j, k]
    return drag_flux(vᵢ, horizontal_speed(uᵢ, vᵢ), p)
end

@inline function u_immersed_drag(i, j, k, grid, clock, model_fields, p)
    uᵢ = @inbounds model_fields.u[i, j, k]
    vᵢ = ℑxyᶠᶜᵃ(i, j, k, grid, model_fields.v)
    return drag_flux(uᵢ, horizontal_speed(uᵢ, vᵢ), p)
end

@inline function v_immersed_drag(i, j, k, grid, clock, model_fields, p)
    uᵢ = ℑxyᶜᶠᵃ(i, j, k, grid, model_fields.u)
    vᵢ = @inbounds model_fields.v[i, j, k]
    return drag_flux(vᵢ, horizontal_speed(uᵢ, vᵢ), p)
end

# Bottom-stress diagnostics on the native u and v grids. Values are zero away
# from active velocity nodes immediately above the regular or immersed bottom.
@inline function bottom_stress_u(i, j, k, grid, u, v, p)
    active = !inactive_node(i, j, k, grid, Face(), Center(), Center())
    above_bottom = k == 1 || inactive_node(i, j, k - 1, grid, Face(), Center(), Center())
    uᵢ = @inbounds u[i, j, k]
    vᵢ = ℑxyᶠᶜᵃ(i, j, k, grid, v)
    stress = drag_flux(uᵢ, horizontal_speed(uᵢ, vᵢ), p)
    return ifelse(active & above_bottom, stress, zero(grid))
end

@inline function bottom_stress_v(i, j, k, grid, u, v, p)
    active = !inactive_node(i, j, k, grid, Center(), Face(), Center())
    above_bottom = k == 1 || inactive_node(i, j, k - 1, grid, Center(), Face(), Center())
    uᵢ = ℑxyᶜᶠᵃ(i, j, k, grid, u)
    vᵢ = @inbounds v[i, j, k]
    stress = drag_flux(vᵢ, horizontal_speed(uᵢ, vᵢ), p)
    return ifelse(active & above_bottom, stress, zero(grid))
end

u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=parameters)
v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=parameters)
u_immersed_drag_bc = FluxBoundaryCondition(u_immersed_drag, discrete_form=true, parameters=parameters)
v_immersed_drag_bc = FluxBoundaryCondition(v_immersed_drag, discrete_form=true, parameters=parameters)

u_stress_bc = FluxBoundaryCondition(u_stress; parameters)
b_relax_bc  = FluxBoundaryCondition(buoyancy_relaxation, discrete_form=true, parameters=parameters)

if slope_config != NoSlopes
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
boundary_layer_closure       = RiBasedVerticalDiffusivity()
vertical_diffusive_closure   = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ = 3e-5, ν = 5e-4)
const biharmonic_damping_timescale = 5days

@inline function horizontal_grid_scale_squared(i, j, k, grid, ℓx, ℓy, ℓz)
    Δx² = Δx(i, j, k, grid, ℓx, ℓy, ℓz)^2
    Δy² = Δy(i, j, k, grid, ℓx, ℓy, ℓz)^2
    return inv(inv(Δx²) + inv(Δy²))
end

@inline function grid_scale_biharmonic_viscosity(i, j, k, grid, ℓx, ℓy, ℓz, clock, model_fields)
    Δh² = horizontal_grid_scale_squared(i, j, k, grid, ℓx, ℓy, ℓz)
    return Δh²^2 / biharmonic_damping_timescale
end

horizontal_biharmonic_closure =
    HorizontalScalarBiharmonicDiffusivity(ν = grid_scale_biharmonic_viscosity,
                                          discrete_form = true)

closures = (boundary_layer_closure,
            vertical_diffusive_closure,
            horizontal_biharmonic_closure)

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

wizard = TimeStepWizard(cfl = 0.5,
                        max_change = 1.05,
                        min_change = 0.0,
                        max_Δt = 30minutes)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2))

# add progress callback
wall_clock = [time_ns()]

function progress(sim)
    message = @sprintf("[%05.2f%%] i: %d, t: %s, max(u): (%6.2e, %6.2e, %6.2e) m s⁻¹, Δt: %s, CFL: %.2f, wall time: %s",
                       100 * (sim.model.clock.time / sim.stop_time),
                       sim.model.clock.iteration,
                       prettytime(sim.model.clock.time),
                       maximum(abs, sim.model.velocities.u),
                       maximum(abs, sim.model.velocities.v),
                       maximum(abs, sim.model.velocities.w),
                       prettytime(sim.Δt),
                       AdvectiveCFL(sim.Δt)(sim.model),
                       prettytime(1e-9 * (time_ns() - wall_clock[1])))

    println(stderr, message)
    flush(stderr)

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, TimeInterval(7days))

# ## Output
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

bottom_stress_u_operation = KernelFunctionOperation{Face, Center, Center}(bottom_stress_u, grid, u, v, parameters)
bottom_stress_v_operation = KernelFunctionOperation{Center, Face, Center}(bottom_stress_v, grid, u, v, parameters)
bottom_stress_u_field = Field(bottom_stress_u_operation)
bottom_stress_v_field = Field(bottom_stress_v_operation)

monthly_outputs = (u = u,
                   v = v,
                   b = b,
                   bottom_stress_u = bottom_stress_u_field,
                   bottom_stress_v = bottom_stress_v_field)

simulation.output_writers[:monthly_means] =
    JLD2Writer(model, monthly_outputs,
               schedule = AveragedTimeInterval(model_month, window = model_month),
               filename = "double_gyre_monthly_mean",
               overwrite_existing = true)

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


# Plot the barotropic circulation derived from the monthly-mean 3D fields

filename_monthly_mean = "double_gyre_monthly_mean.jld2"

U_timeseries = FieldTimeSeries(filename_monthly_mean, "u"; grid, architecture = CPU())
V_timeseries = FieldTimeSeries(filename_monthly_mean, "v"; grid, architecture = CPU())

# time-average; adjust accordingly to avoid spinup
U_mean = Field{Oceananigans.Fields.location(U_timeseries)...}(on_architecture(CPU(), grid))
V_mean = Field{Oceananigans.Fields.location(V_timeseries)...}(on_architecture(CPU(), grid))

for (iter, time_snapshop) in enumerate(round(Int, length(U_timeseries)/2):length(U_timeseries))
    parent(U_mean) .= parent(U_mean) * (iter - 1) / iter .+ parent(U_timeseries[time_snapshop]) / iter
    parent(V_mean) .= parent(V_mean) * (iter - 1) / iter .+ parent(V_timeseries[time_snapshop]) / iter
end

U_barotropic_mean = Field(Average(U_mean, dims = 3))
V_barotropic_mean = Field(Average(V_mean, dims = 3))
compute!(U_barotropic_mean)
compute!(V_barotropic_mean)

fig = Figure(size = (1650, 1250))

title_U = "Depth- and Time-Averaged Zonal Velocity"
ax_U = Axis(fig[1:2, 1]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)")
hm_U = heatmap!(ax_U, λᵤ, φᵤ, U_barotropic_mean; colorrange = ulims ./10, colormap = :balance)
Colorbar(fig[1:2, 2], hm_U)

title_V = "Depth- and Time-Averaged Meridional Velocity"
ax_V = Axis(fig[3:4, 1]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)")
hm_V = heatmap!(ax_V, λᵥ, φᵥ, V_barotropic_mean; colorrange = vlims ./10, colormap = :balance)
Colorbar(fig[3:4, 2], hm_V)

Ψ = CumulativeIntegral(-U_barotropic_mean, dims = 2) |> Field
Ψlims = extrema(Ψ) .* extrema_reduction_factor

title_Ψ = "Barotropic Streamfunction"
ax_Ψ = Axis(fig[2:3, 3]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)")
hm_Ψ = heatmap!(ax_Ψ, λᵤ, φᵤ, Ψ; colorrange = Ψlims, colormap = :balance)
Colorbar(fig[2:3, 4], hm_Ψ)

save("double_gyre_circulation.png", fig)
