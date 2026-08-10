#!/usr/bin/env julia

"""
Convert the online depth-integrated momentum budget to a barotropic
vorticity budget by applying Oceananigans' native, metric-aware C-grid curl.

Usage
-----

    julia --project=. generate_vorticity_budget/convert_barotropic_vorticity_budget.jl \
        [momentum_budget.jld2] [budget_state.jld2] [output.jld2] \
        [--time-resolution=yearly|monthly] [--force]

The state file is optional. When it is present, the output also contains the
curl of the barotropic transport and its finite-difference tendency over each
averaging window. The default `yearly` mode time-weights the source monthly
means and writes one record per model year. Use `--time-resolution=monthly`
to retain every source record.

Sign convention
---------------

The momentum diagnostics obey

    total = advection + coriolis + pressure + closure + immersed_stress

and regular FluxBoundaryConditions enter the full tendency as

    full_rhs = total + regular_bottom - surface_flux.

`immersed_bottom` is a diagnostic subset of `immersed_stress`; it must not be
added to `full_rhs` a second time. For a decomposition with a single bottom
drag term, this script also saves

    bottom_drag = regular_bottom + immersed_bottom
    immersed_stress_remainder = immersed_stress - immersed_bottom.

Khatri et al. (2024) terms
---------------------------

When the state file provides `transport_v` and `free_surface` and the model
Coriolis is `HydrostaticSphericalCoriolis`, this script also assembles the
barotropic vorticity budget of Khatri et al. (2024, JAMES, Eq. 3/A17):

    beta_V = J(pb, H)/ρo + ẑ⋅(∇∧τs/ρo - ∇∧τb/ρo + ∇∧A + ∇∧B)
             - f Qm/ρo + f ∂tη - ẑ⋅(∇∧Ut)

`beta_V` is computed directly from the saved transport (β interpolated to the
vorticity point times V). Following their Appendix B2, the raw curls of the
Coriolis and pressure-gradient terms are individually corrupted by C-grid
numerical noise, so `bottom_pressure_torque` (J(pb,H)/ρo) is instead diagnosed
as the Eq. B4 combination. For a split-explicit free surface, the total pressure
torque must include the effective external-mode pressure contribution applied
during the fast subcycles:

    bottom_pressure_torque = coriolis + pressure + split_explicit_pressure
                             + beta_V - eta_tendency_term

The Khatri closure term also includes `immersed_stress_remainder` and
`slow_decomposition_residual`, which are required to reproduce the exact signed
momentum tendency used by the solver. This assumes Qm = 0 (this model has no
surface mass flux).
"""

using Oceananigans
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis, fᶠᶠᵃ
using Oceananigans.Fields: location
using Oceananigans.Grids: hack_cosd, φnode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Operators: ζ₃ᶠᶠᶜ, ℑxᶠᵃᵃ, ℑxyᶠᶠᵃ
using JLD2
using Printf

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTPUT_DIR = joinpath(PROJECT_ROOT, "outputs")
const DEFAULT_BUDGET = joinpath(OUTPUT_DIR, "double_gyre_barotropic_budget.jld2")
const DEFAULT_STATE  = joinpath(OUTPUT_DIR, "double_gyre_barotropic_budget_state.jld2")
const DEFAULT_OUTPUT = joinpath(OUTPUT_DIR, "double_gyre_barotropic_vorticity_budget.jld2")
const DEFAULT_MONTHLY_OUTPUT = joinpath(OUTPUT_DIR, "double_gyre_barotropic_vorticity_budget_monthly.jld2")
const MODEL_YEAR = 365 * 86400.0

const SOURCE_PAIRS = (
    advection       = ("advection_u",       "advection_v"),
    coriolis        = ("coriolis_u",        "coriolis_v"),
    pressure        = ("pressure_u",        "pressure_v"),
    closure         = ("closure_u",         "closure_v"),
    immersed_stress = ("immersed_stress_u", "immersed_stress_v"),
    total           = ("total_u",           "total_v"),
    regular_bottom  = ("regular_bottom_u",  "regular_bottom_v"),
    immersed_bottom = ("immersed_bottom_u", "immersed_bottom_v"),
    surface_flux    = ("surface_u",         "surface_v"),
)

const OPTIONAL_DISCRETE_PAIRS = (
    transport_tendency            = ("transport_tendency_u",            "transport_tendency_v"),
    solver_slow                   = ("solver_slow_u",                   "solver_slow_v"),
    component_rhs                 = ("component_rhs_u",                 "component_rhs_v"),
    slow_decomposition_residual   = ("slow_decomposition_residual_u",   "slow_decomposition_residual_v"),
    split_explicit_pressure       = ("split_explicit_pressure_u",       "split_explicit_pressure_v"),
    closed_rhs                    = ("closed_rhs_u",                    "closed_rhs_v"),
    closure_residual              = ("closure_residual_u",              "closure_residual_v"),
)

function command_line()
    force = "--force" in ARGS
    resolution_args = filter(arg -> startswith(arg, "--time-resolution="), ARGS)
    length(resolution_args) <= 1 || error("Specify --time-resolution only once")
    resolution_name = isempty(resolution_args) ? "yearly" : split(only(resolution_args), "="; limit=2)[2]
    resolution_name in ("yearly", "monthly") ||
        error("Invalid time resolution '$resolution_name'; expected yearly or monthly")
    time_resolution = Symbol(resolution_name)

    positional = filter(arg -> arg != "--force" && !startswith(arg, "--time-resolution="), ARGS)
    length(positional) <= 3 ||
        error("Expected at most three file arguments plus --time-resolution and optional --force")

    budget = abspath(get(positional, 1, DEFAULT_BUDGET))
    state  = abspath(get(positional, 2, DEFAULT_STATE))
    default_output = time_resolution == :yearly ? DEFAULT_OUTPUT : DEFAULT_MONTHLY_OUTPUT
    output = abspath(get(positional, 3, default_output))
    return (; budget, state, output, force, time_resolution)
end

"""Group source records into output averaging periods."""
function output_record_groups(times, time_resolution)
    time_resolution == :monthly && return [[record] for record in eachindex(times)]

    length(times) >= 2 || error("Yearly conversion requires at least two source records")
    initial_time = first(times)
    groups = Vector{Vector{Int}}()

    for record in 2:length(times)
        elapsed = times[record] - initial_time
        year_index = max(1, ceil(Int, elapsed / MODEL_YEAR - 100eps(Float64)))
        while length(groups) < year_index
            push!(groups, Int[])
        end
        push!(groups[year_index], record)
    end

    filter!(!isempty, groups)
    return groups
end

function iterations_and_times(path)
    jldopen(path, "r") do file
        iterations = sort(parse.(Int, collect(keys(file["timeseries/t"]))))
        times = Float64[file["timeseries/t/$iteration"] for iteration in iterations]
        return iterations, times
    end
end

function available_timeseries(path)
    return jldopen(path, "r") do file
        Set(String.(collect(keys(file["timeseries"]))))
    end
end

horizontal_grid(grid) = grid
horizontal_grid(grid::ImmersedBoundaryGrid) = grid.underlying_grid

"""
Return the native spherical C-grid curl as a two-dimensional Field.

Depth-integrated terms exist over every horizontal water column and have no
vertical index. Use the underlying horizontal grid so ImmersedBoundaryGrid
conditional differences do not incorrectly mask slope columns at `k=1`.
"""
function native_curl(u, v)
    grid = horizontal_grid(u.grid)
    operation = KernelFunctionOperation{Face, Face, Nothing}(ζ₃ᶠᶠᶜ, grid, u, v)
    curl = Field(operation)
    compute!(curl)
    return curl
end

raw_array(field) = Array(parent(field))

# β at the vorticity (Face, Face) point, i.e. df/dy = 2Ω cos(φ)/R.
@inline function betaᶠᶠᵃ(i, j, k, grid, coriolis::HydrostaticSphericalCoriolis)
    φ = φnode(j, grid, Face())
    return 2 * coriolis.rotation_rate * hack_cosd(φ) / grid.radius
end

@inline function beta_times_transport(i, j, k, grid, coriolis, v)
    return betaᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, v)
end

@inline function f_eta_tendency(i, j, k, grid, coriolis, η_new, η_old, inv_Δt)
    Δη = ℑxyᶠᶠᵃ(i, j, k, grid, η_new) - ℑxyᶠᶠᵃ(i, j, k, grid, η_old)
    return fᶠᶠᵃ(i, j, k, grid, coriolis) * Δη * inv_Δt
end

"""β times the vertically integrated meridional transport, interpolated to the vorticity point."""
function compute_beta_V(grid, coriolis, v_field)
    operation = KernelFunctionOperation{Face, Face, Nothing}(beta_times_transport, grid, coriolis, v_field)
    field = Field(operation)
    compute!(field)
    return raw_array(field)
end

"""f ∂tη over [η_old, η_new], interpolated to the vorticity point."""
function compute_f_eta_tendency(grid, coriolis, η_new_field, η_old_field, Δt)
    η_new_2d = surface_field_2d(grid, η_new_field)
    η_old_2d = surface_field_2d(grid, η_old_field)
    operation = KernelFunctionOperation{Face, Face, Nothing}(f_eta_tendency, grid, coriolis, η_new_2d, η_old_2d, 1 / Δt)
    field = Field(operation)
    compute!(field)
    return raw_array(field)
end

# η is saved at (Center, Center, Face) (the free-surface's own vertical location);
# reinterpret it as a genuine 2-D (Center, Center, Nothing) field before curl-point interpolation.
function surface_field_2d(grid, field3d)
    field2d = Field{Center, Center, Nothing}(grid)
    interior(field2d, :, :, 1) .= interior(field3d, :, :, 1)
    fill_halo_regions!(field2d)
    return field2d
end

function install_field_metadata!(file, name, field)
    prefix = "timeseries/$name/serialized"
    file["$prefix/location"] = location(field)
    file["$prefix/indices"] = (:, :, :)
    file["$prefix/boundary_conditions"] = field.boundary_conditions
    return nothing
end

function write_record!(file, name, iteration, array)
    file["timeseries/$name/$iteration"] = array
    return nothing
end

function write_description!(file, name, description; units="m s⁻²")
    file["metadata/terms/$name/description"] = description
    file["metadata/terms/$name/units"] = units
    return nothing
end

function load_series(path, names, iterations)
    return Dict(name => FieldTimeSeries(path, name;
                                        architecture=CPU(),
                                        backend=OnDisk(),
                                        iterations=iterations)
                for name in names)
end

function source_curls(series, active_pairs, record)
    curls = Dict{Symbol, Array{Float64, 3}}()
    template_field = nothing

    for (term, (u_name, v_name)) in active_pairs
        curl_field = native_curl(series[u_name][record], series[v_name][record])
        template_field === nothing && (template_field = curl_field)
        curls[term] = raw_array(curl_field)
    end

    return curls, template_field
end

function averaged_source_curls(series, active_pairs, records, times, time_resolution)
    if time_resolution == :monthly
        return source_curls(series, active_pairs, only(records))
    end

    durations = [times[record] - times[record - 1] for record in records]
    total_duration = sum(durations)
    total_duration > 0 || error("Non-positive yearly averaging duration")

    averaged = nothing
    template_field = nothing
    for (record, duration) in zip(records, durations)
        curls, record_template = source_curls(series, active_pairs, record)
        weight = duration / total_duration

        if averaged === nothing
            averaged = Dict(term => weight .* array for (term, array) in curls)
            template_field = record_template
        else
            for (term, array) in curls
                averaged[term] .+= weight .* array
            end
        end
    end

    return averaged, template_field
end

function add_derived_curls!(curls)
    curls[:wind_stress] = -curls[:surface_flux]
    curls[:bottom_drag] = curls[:regular_bottom] + curls[:immersed_bottom]
    curls[:immersed_stress_remainder] = curls[:immersed_stress] - curls[:immersed_bottom]
    curls[:core_rhs] = (curls[:advection] + curls[:coriolis] + curls[:pressure] +
                        curls[:closure] + curls[:immersed_stress])
    curls[:core_closure_residual] = curls[:total] - curls[:core_rhs]
    curls[:full_rhs] = curls[:total] + curls[:regular_bottom] - curls[:surface_flux]
    curls[:decomposed_full_rhs] = (curls[:advection] + curls[:coriolis] + curls[:pressure] +
                                   curls[:closure] + curls[:bottom_drag] +
                                   curls[:immersed_stress_remainder] + curls[:wind_stress])
    curls[:full_decomposition_residual] = curls[:full_rhs] - curls[:decomposed_full_rhs]
    return nothing
end

function main()
    args = command_line()
    isfile(args.budget) || error("Momentum-budget file not found: $(args.budget)")
    isfile(args.output) && !args.force &&
        error("Output already exists: $(args.output). Pass --force to replace it.")

    iterations, times = iterations_and_times(args.budget)
    available = available_timeseries(args.budget)
    active_pairs = collect(pairs(SOURCE_PAIRS))
    for (term, pair) in pairs(OPTIONAL_DISCRETE_PAIRS)
        all(name -> name in available, pair) && push!(active_pairs, term => pair)
    end

    all_source_names = reduce(vcat, [collect(pair) for (_, pair) in active_pairs])
    series = load_series(args.budget, all_source_names, iterations)
    grid = horizontal_grid(first(values(series)).grid)
    record_groups = output_record_groups(times, args.time_resolution)

    temporary_output = args.output * ".partial"
    isfile(temporary_output) && rm(temporary_output; force=true)

    @info "Converting momentum budget to vorticity budget" source=args.budget output=args.output time_resolution=args.time_resolution source_records=length(iterations) output_records=length(record_groups)

    # Retained for comparison with the endpoint-derived transport tendency.
    full_rhs_by_iteration = Dict{Int, Array{Float64, 3}}()

    # Retained to assemble the Khatri et al. (2024) bottom-pressure-torque diagnostic (Eq. B4).
    curls_by_iteration = Dict{Int, Dict{Symbol, Array{Float64, 3}}}()

    try
        jldopen(temporary_output, "w") do output
            output["metadata/format"] = "Oceananigans barotropic vorticity budget"
            output["metadata/version"] = 1
            output["metadata/source_momentum_budget"] = args.budget
            output["metadata/source_budget_state"] = isfile(args.state) ? args.state : ""
            output["metadata/time_resolution"] = string(args.time_resolution)
            output["metadata/curl_operator"] = "Oceananigans.Operators.ζ₃ᶠᶠᶜ (native metric-aware C-grid curl)"
            output["metadata/horizontal_grid_note"] = "Curl uses the underlying horizontal grid; depth-integrated 2-D fields must not be masked using immersed-bottom activity at k=1."
            output["metadata/sign_convention"] = "full_rhs = total + regular_bottom - surface_flux; immersed_bottom is already included in immersed_stress"
            output["metadata/initial_record_note"] = args.time_resolution == :monthly ?
                "The t=0 record is initialization output, not an averaged month." :
                "Yearly output omits the t=0 initialization record."
            output["serialized/grid"] = grid

            write_description!(output, "advection", "Curl of the signed, depth-integrated nonlinear momentum-advection tendency")
            write_description!(output, "coriolis", "Curl of the signed, depth-integrated Coriolis tendency")
            write_description!(output, "pressure", "Curl of the combined depth-integrated free-surface, hydrostatic, and grid-slope pressure tendency")
            write_description!(output, "closure", "Curl of the depth-integrated parameterized stress-divergence tendency")
            write_description!(output, "immersed_stress", "Curl of the immersed-boundary stress tendency; includes immersed-bottom drag")
            write_description!(output, "total", "Curl of the saved total slow momentum tendency")
            write_description!(output, "regular_bottom", "Curl of the regular-bottom momentum flux, with its signed +bottom convention")
            write_description!(output, "immersed_bottom", "Curl of immersed-bottom drag alone; diagnostic subset of immersed_stress")
            write_description!(output, "surface_flux", "Curl of the raw top momentum flux; subtract this field in the RHS")
            write_description!(output, "wind_stress", "Signed wind-stress contribution, equal to -surface_flux")
            write_description!(output, "bottom_drag", "Combined regular plus immersed bottom-drag curl")
            write_description!(output, "immersed_stress_remainder", "immersed_stress minus immersed_bottom")
            write_description!(output, "core_rhs", "advection + coriolis + pressure + closure + immersed_stress")
            write_description!(output, "core_closure_residual", "total - core_rhs")
            write_description!(output, "full_rhs", "total + regular_bottom - surface_flux")
            write_description!(output, "decomposed_full_rhs", "advection + coriolis + pressure + closure + bottom_drag + immersed_stress_remainder + wind_stress")
            write_description!(output, "full_decomposition_residual", "full_rhs - decomposed_full_rhs")

            if any(first(pair) == :transport_tendency for pair in active_pairs)
                write_description!(output, "transport_tendency", "Curl of the exact step-to-step barotropic transport tendency")
                write_description!(output, "solver_slow", "Curl of Gⁿ.U/V actually used by the final RK-stage split-explicit solve")
                write_description!(output, "component_rhs", "Curl of stage-used total + regular_bottom - surface_flux")
                write_description!(output, "slow_decomposition_residual", "solver_slow - component_rhs")
                write_description!(output, "split_explicit_pressure", "Exact effective external-pressure torque over the slow step, including fast-subcycle filtering")
                write_description!(output, "closed_rhs", "component_rhs + slow_decomposition_residual + split_explicit_pressure")
                write_description!(output, "closure_residual", "transport_tendency - closed_rhs")
            end

            metadata_installed = false
            template_field = nothing

            for (output_record, records) in enumerate(record_groups)
                curls, record_template = averaged_source_curls(series, active_pairs, records,
                                                                times, args.time_resolution)
                template_field === nothing && (template_field = record_template)
                add_derived_curls!(curls)

                record = last(records)
                iteration = iterations[record]
                time = times[record]
                comparison_rhs = curls[:full_rhs]
                full_rhs_by_iteration[iteration] = comparison_rhs
                curls_by_iteration[iteration] = curls

                if !metadata_installed
                    for name in string.(collect(keys(curls)))
                        install_field_metadata!(output, name, template_field)
                    end
                    longitude, latitude, _ = nodes(template_field)
                    output["coordinates/longitude"] = longitude
                    output["coordinates/latitude"] = latitude
                    metadata_installed = true
                end

                output["timeseries/t/$iteration"] = time
                for (term, array) in curls
                    write_record!(output, string(term), iteration, array)
                end

                @info @sprintf("Converted %s record %d/%d: iteration %d, day %.4f",
                               string(args.time_resolution), output_record,
                               length(record_groups), iteration, time / 86400)
            end

            if isfile(args.state)
                state_iterations, state_times = iterations_and_times(args.state)
                state_iterations == iterations || error("State and budget iterations do not match")
                all(isapprox.(state_times, times; rtol=0, atol=100eps(maximum(times)))) ||
                    error("State and budget timestamps do not match")

                state = load_series(args.state, ["transport_u", "transport_v", "free_surface"], state_iterations)
                write_description!(output, "transport_vorticity", "Curl of the instantaneous depth-integrated transport", units="m s⁻¹")
                write_description!(output, "transport_vorticity_tendency", "Finite-difference transport-vorticity tendency over the preceding averaging interval")
                comparison_name = "full_rhs"
                write_description!(output, "tendency_closure_residual", "transport_vorticity_tendency - $comparison_name")

                install_field_metadata!(output, "transport_vorticity", template_field)
                install_field_metadata!(output, "transport_vorticity_tendency", template_field)
                install_field_metadata!(output, "tendency_closure_residual", template_field)

                coriolis = jldopen(args.state, "r") do file
                    haskey(file, "coriolis/rotation_rate") ?
                        HydrostaticSphericalCoriolis(rotation_rate=file["coriolis/rotation_rate"]) :
                        file["coriolis"]
                end
                is_spherical_coriolis = coriolis isa HydrostaticSphericalCoriolis
                first_record_curls = first(values(curls_by_iteration))
                has_khatri_solver_terms = all(haskey(first_record_curls, term)
                                              for term in (:split_explicit_pressure,
                                                           :slow_decomposition_residual))
                can_compute_khatri_budget = is_spherical_coriolis && has_khatri_solver_terms

                if can_compute_khatri_budget
                    write_description!(output, "beta_V", "β (Khatri et al. 2024 Eq. 3) times the vertically integrated meridional transport, interpolated to the vorticity point")
                    write_description!(output, "eta_tendency_term", "f ∂tη, the free-surface tendency term in Khatri et al. (2024) Eq. 3")
                    write_description!(output, "bottom_pressure_torque", "J(pb,H)/ρo, diagnosed as coriolis + pressure + split_explicit_pressure + beta_V - eta_tendency_term following Khatri et al. (2024) Eq. B4; assumes Qm=0")
                    write_description!(output, "khatri_budget_residual", "beta_V - (bottom_pressure_torque + wind_stress + bottom_drag + advection + closure + immersed_stress_remainder + slow_decomposition_residual + eta_tendency_term - transport_vorticity_tendency); validates Khatri et al. (2024) Eq. 3 closure with the complete split-explicit solver tendency")
                    install_field_metadata!(output, "beta_V", template_field)
                    install_field_metadata!(output, "eta_tendency_term", template_field)
                    install_field_metadata!(output, "bottom_pressure_torque", template_field)
                    install_field_metadata!(output, "khatri_budget_residual", template_field)
                elseif !is_spherical_coriolis
                    @warn "Coriolis is not HydrostaticSphericalCoriolis; skipping Khatri et al. (2024) beta_V/bottom_pressure_torque diagnostics" coriolis=typeof(coriolis)
                else
                    @warn "Split-explicit solver diagnostics are incomplete; skipping Khatri et al. (2024) diagnostics rather than writing a non-closing budget" required_terms=(:split_explicit_pressure, :slow_decomposition_residual)
                end

                for records in record_groups
                    record = last(records)
                    iteration = state_iterations[record]
                    time = state_times[record]
                    transport_vorticity = raw_array(native_curl(state["transport_u"][record],
                                                                state["transport_v"][record]))
                    write_record!(output, "transport_vorticity", iteration, transport_vorticity)

                    beta_V = can_compute_khatri_budget ? compute_beta_V(grid, coriolis, state["transport_v"][record]) : nothing
                    can_compute_khatri_budget && write_record!(output, "beta_V", iteration, beta_V)

                    if record == 1
                        # FieldTimeSeries expects every field to have the global
                        # iterations. No preceding interval exists at t=0, so NaN
                        # is the honest sentinel for the undefined tendency.
                        undefined_tendency = fill(NaN, size(transport_vorticity))
                        write_record!(output, "transport_vorticity_tendency", iteration, undefined_tendency)
                        write_record!(output, "tendency_closure_residual", iteration, undefined_tendency)

                        if can_compute_khatri_budget
                            write_record!(output, "eta_tendency_term", iteration, undefined_tendency)
                            write_record!(output, "bottom_pressure_torque", iteration, undefined_tendency)
                            write_record!(output, "khatri_budget_residual", iteration, undefined_tendency)
                        end
                    else
                        previous_record = args.time_resolution == :monthly ? record - 1 : first(records) - 1
                        previous_time = state_times[previous_record]
                        previous_vorticity = raw_array(native_curl(state["transport_u"][previous_record],
                                                                   state["transport_v"][previous_record]))
                        tendency = (transport_vorticity - previous_vorticity) / (time - previous_time)
                        residual = tendency - full_rhs_by_iteration[iteration]
                        write_record!(output, "transport_vorticity_tendency", iteration, tendency)
                        write_record!(output, "tendency_closure_residual", iteration, residual)

                        if can_compute_khatri_budget
                            eta_tendency_term = compute_f_eta_tendency(grid, coriolis,
                                                                       state["free_surface"][record],
                                                                       state["free_surface"][previous_record],
                                                                       time - previous_time)
                            write_record!(output, "eta_tendency_term", iteration, eta_tendency_term)

                            record_curls = curls_by_iteration[iteration]
                            bottom_pressure_torque = (record_curls[:coriolis] .+ record_curls[:pressure] .+
                                                      record_curls[:split_explicit_pressure] .+ beta_V .-
                                                      eta_tendency_term)
                            write_record!(output, "bottom_pressure_torque", iteration, bottom_pressure_torque)

                            khatri_rhs = (bottom_pressure_torque .+ record_curls[:wind_stress] .+ record_curls[:bottom_drag] .+
                                         record_curls[:advection] .+ record_curls[:closure] .+
                                         record_curls[:immersed_stress_remainder] .+
                                         record_curls[:slow_decomposition_residual] .+
                                         eta_tendency_term .- tendency)
                            khatri_budget_residual = beta_V .- khatri_rhs
                            write_record!(output, "khatri_budget_residual", iteration, khatri_budget_residual)
                        end
                    end
                end
            else
                @warn "State file not found; transport-vorticity tendency was not written" state=args.state
            end
        end

        isfile(args.output) && rm(args.output; force=true)
        mv(temporary_output, args.output)
    catch
        isfile(temporary_output) && rm(temporary_output; force=true)
        rethrow()
    end

    @info "Vorticity-budget conversion complete" output=args.output size=filesize(args.output)
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
