# Online depth-integrated momentum diagnostics for a barotropic vorticity budget.
# Terms use the same discrete operators as HydrostaticFreeSurfaceModel.

using Oceananigans.Advection: U_dot_∇u, U_dot_∇v
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U
using Oceananigans.Fields: immersed_boundary_condition, location
using Oceananigans.ImmersedBoundaries: peripheral_node
using Oceananigans.Models.HydrostaticFreeSurfaceModels: explicit_barotropic_pressure_x_gradient,
                                                        explicit_barotropic_pressure_y_gradient,
                                                        grid_slope_contribution_x,
                                                        grid_slope_contribution_y
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, Δz
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ,
                                       immersed_∂ⱼ_τ₁ⱼ, immersed_∂ⱼ_τ₂ⱼ

@inline budget_advection_u(i, j, k, grid, advection, velocities) = -U_dot_∇u(i, j, k, grid, advection, velocities)
@inline budget_advection_v(i, j, k, grid, advection, velocities) = -U_dot_∇v(i, j, k, grid, advection, velocities)
@inline budget_coriolis_u(i, j, k, grid, coriolis, velocities) = -x_f_cross_U(i, j, k, grid, coriolis, velocities)
@inline budget_coriolis_v(i, j, k, grid, coriolis, velocities) = -y_f_cross_U(i, j, k, grid, coriolis, velocities)

@inline function budget_pressure_u(i, j, k, grid, free_surface, pHY′, buoyancy, vertical_coordinate, model_fields)
    return (-explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface)
            - ∂xᶠᶜᶜ(i, j, k, grid, pHY′)
            - grid_slope_contribution_x(i, j, k, grid, buoyancy, vertical_coordinate, model_fields))
end

@inline function budget_pressure_v(i, j, k, grid, free_surface, pHY′, buoyancy, vertical_coordinate, model_fields)
    return (-explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface)
            - ∂yᶜᶠᶜ(i, j, k, grid, pHY′)
            - grid_slope_contribution_y(i, j, k, grid, buoyancy, vertical_coordinate, model_fields))
end

@inline budget_closure_u(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy) =
    -∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy)
@inline budget_closure_v(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy) =
    -∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy)
@inline budget_immersed_stress_u(i, j, k, grid, velocities, immersed_bc, closure, diffusivities, clock, model_fields) =
    -immersed_∂ⱼ_τ₁ⱼ(i, j, k, grid, velocities, immersed_bc, closure, diffusivities, clock, model_fields)
@inline budget_immersed_stress_v(i, j, k, grid, velocities, immersed_bc, closure, diffusivities, clock, model_fields) =
    -immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid, velocities, immersed_bc, closure, diffusivities, clock, model_fields)
@inline budget_total(i, j, k, grid, G) = @inbounds G[i, j, k]

@inline function depth_integrated_u(i, j, k, grid, term, args...)
    integral = zero(grid)
    for n in 1:grid.Nz
        inactive = peripheral_node(i, j, n, grid, Face(), Center(), Center())
        value = term(i, j, n, grid, args...)
        integral += Δz(i, j, n, grid, Face(), Center(), Center()) * ifelse(inactive, zero(grid), value)
    end
    return integral
end

@inline function depth_integrated_v(i, j, k, grid, term, args...)
    integral = zero(grid)
    for n in 1:grid.Nz
        inactive = peripheral_node(i, j, n, grid, Center(), Face(), Center())
        value = term(i, j, n, grid, args...)
        integral += Δz(i, j, n, grid, Center(), Face(), Center()) * ifelse(inactive, zero(grid), value)
    end
    return integral
end

# FluxBoundaryCondition contributes +Q_bottom and -Q_top to the vertical integral.
# Immersed-bottom fluxes are already present in immersed_stress, so they are
# saved separately and must not be added a second time.
@inline function regular_bottom_flux_u(i, j, k, grid, u, v, p)
    active = !inactive_node(i, j, 1, grid, Face(), Center(), Center())
    uᵢ = @inbounds u[i, j, 1]
    vᵢ = ℑxyᶠᶜᵃ(i, j, 1, grid, v)
    return ifelse(active, drag_flux(uᵢ, horizontal_speed(uᵢ, vᵢ), p), zero(grid))
end

@inline function regular_bottom_flux_v(i, j, k, grid, u, v, p)
    active = !inactive_node(i, j, 1, grid, Center(), Face(), Center())
    uᵢ = ℑxyᶜᶠᵃ(i, j, 1, grid, u)
    vᵢ = @inbounds v[i, j, 1]
    return ifelse(active, drag_flux(vᵢ, horizontal_speed(uᵢ, vᵢ), p), zero(grid))
end

@inline function immersed_bottom_flux_u(i, j, k, grid, u, v, p)
    flux = zero(grid)
    for n in 2:grid.Nz
        active = !inactive_node(i, j, n, grid, Face(), Center(), Center())
        above_bottom = inactive_node(i, j, n - 1, grid, Face(), Center(), Center())
        uᵢ = @inbounds u[i, j, n]
        vᵢ = ℑxyᶠᶜᵃ(i, j, n, grid, v)
        flux += ifelse(active & above_bottom, drag_flux(uᵢ, horizontal_speed(uᵢ, vᵢ), p), zero(grid))
    end
    return flux
end

@inline function immersed_bottom_flux_v(i, j, k, grid, u, v, p)
    flux = zero(grid)
    for n in 2:grid.Nz
        active = !inactive_node(i, j, n, grid, Center(), Face(), Center())
        above_bottom = inactive_node(i, j, n - 1, grid, Center(), Face(), Center())
        uᵢ = ℑxyᶜᶠᵃ(i, j, n, grid, u)
        vᵢ = @inbounds v[i, j, n]
        flux += ifelse(active & above_bottom, drag_flux(vᵢ, horizontal_speed(uᵢ, vᵢ), p), zero(grid))
    end
    return flux
end

@inline function surface_flux_u(i, j, k, grid, clock, p)
    φ = φnode(j, grid, Center())
    return u_stress(zero(grid), φ, clock.time, p)
end
@inline surface_flux_v(i, j, k, grid) = zero(grid)

"""
Construct online two-dimensional momentum-budget fields. The closure identity is
`total = advection + coriolis + pressure + closure + immersed_stress`.
Regular boundary fluxes enter the full slow tendency as `+bottom - surface`.
"""
function barotropic_budget_outputs(model, parameters)
    grid = model.grid
    u, v, _ = model.velocities
    velocities = model.velocities
    model_fields = fields(model)

    uop(term, args...) = KernelFunctionOperation{Face, Center, Nothing}(depth_integrated_u, grid, term, args...)
    vop(term, args...) = KernelFunctionOperation{Center, Face, Nothing}(depth_integrated_v, grid, term, args...)

    advection_u = Field(uop(budget_advection_u, model.advection.momentum, velocities))
    advection_v = Field(vop(budget_advection_v, model.advection.momentum, velocities))
    coriolis_u = Field(uop(budget_coriolis_u, model.coriolis, velocities))
    coriolis_v = Field(vop(budget_coriolis_v, model.coriolis, velocities))

    pressure_args = (model.free_surface, model.pressure.pHY′, model.buoyancy,
                     model.vertical_coordinate, model_fields)
    pressure_u = Field(uop(budget_pressure_u, pressure_args...))
    pressure_v = Field(vop(budget_pressure_v, pressure_args...))

    closure_args = (model.closure, model.diffusivity_fields, model.clock, model_fields, model.buoyancy)
    closure_u = Field(uop(budget_closure_u, closure_args...))
    closure_v = Field(vop(budget_closure_v, closure_args...))

    immersed_args_u = (velocities, immersed_boundary_condition(u), model.closure,
                       model.diffusivity_fields, model.clock, model_fields)
    immersed_args_v = (velocities, immersed_boundary_condition(v), model.closure,
                       model.diffusivity_fields, model.clock, model_fields)
    immersed_stress_u = Field(uop(budget_immersed_stress_u, immersed_args_u...))
    immersed_stress_v = Field(vop(budget_immersed_stress_v, immersed_args_v...))
    total_u = Field(uop(budget_total, model.timestepper.Gⁿ.u))
    total_v = Field(vop(budget_total, model.timestepper.Gⁿ.v))

    regular_bottom_u = Field(KernelFunctionOperation{Face, Center, Nothing}(regular_bottom_flux_u, grid, u, v, parameters))
    regular_bottom_v = Field(KernelFunctionOperation{Center, Face, Nothing}(regular_bottom_flux_v, grid, u, v, parameters))
    immersed_bottom_u = Field(KernelFunctionOperation{Face, Center, Nothing}(immersed_bottom_flux_u, grid, u, v, parameters))
    immersed_bottom_v = Field(KernelFunctionOperation{Center, Face, Nothing}(immersed_bottom_flux_v, grid, u, v, parameters))
    surface_u = Field(KernelFunctionOperation{Face, Center, Nothing}(surface_flux_u, grid, model.clock, parameters))
    surface_v = Field(KernelFunctionOperation{Center, Face, Nothing}(surface_flux_v, grid))

    return (; advection_u, advection_v, coriolis_u, coriolis_v,
              pressure_u, pressure_v, closure_u, closure_v,
              immersed_stress_u, immersed_stress_v, total_u, total_v,
              regular_bottom_u, regular_bottom_v,
              immersed_bottom_u, immersed_bottom_v, surface_u, surface_v)
end

# A discrete transport budget for SplitRungeKutta3 + SplitExplicitFreeSurface.
#
# During the final RK stage, the slow transport forcing is constructed from
# the stage-2 three-dimensional tendencies. A TendencyCallsite callback copies
# the individual stage-2 terms below. After the completed time step, Gⁿ.U/V
# still contains the vertically integrated slow forcing actually passed to the
# split-explicit solver. Comparing that forcing with the exact step-to-step
# transport increment diagnoses the effective, subcycle-averaged external
# pressure contribution without needing to save every fast substep.

mutable struct DiscreteTransportBudget{S, O, U, V}
    source_terms :: S
    outputs :: O
    previous_u :: U
    previous_v :: V
    previous_time :: Float64
    initialized :: Bool
    stage_two_captured :: Bool
end

"""
Diagnostic wrapper that updates the discrete transport budget before
`WindowedTimeAverage` diagnostics sample its fields. Diagnostics are evaluated
in insertion order after each completed model time step.
"""
struct DiscreteTransportBudgetDiagnostic{B, S} <: Oceananigans.AbstractDiagnostic
    budget :: B
    schedule :: S
end

function zero_field_like(field)
    LX, LY, LZ = location(field)
    result = Field{LX, LY, LZ}(field.grid;
                               boundary_conditions = field.boundary_conditions)
    parent(result) .= 0
    return result
end

function copy_interior_and_fill_halos!(destination, source)
    interior(destination) .= interior(source)
    fill_halo_regions!(destination)
    return nothing
end

function zero_interior_and_fill_halos!(field)
    interior(field) .= 0
    fill_halo_regions!(field)
    return nothing
end

function combine_fields!(destination, fields_and_signs...)
    zero_interior_and_fill_halos!(destination)
    destination_interior = interior(destination)
    for (field, sign) in fields_and_signs
        destination_interior .+= sign .* interior(field)
    end
    fill_halo_regions!(destination)
    return nothing
end

"""
    discrete_barotropic_transport_budget(model, parameters)

Construct storage and callbacks for a discrete, term-by-term depth-integrated
transport budget. The returned fields obey

`transport_tendency = component_rhs + slow_decomposition_residual + split_explicit_pressure`

at every slow time step. Here `split_explicit_pressure` is the exact effective
external-mode contribution over a complete slow step, including the solver's
fast-subcycle averaging. `slow_decomposition_residual` tests whether the saved
physical slow terms plus regular boundary fluxes reproduce the actual Gⁿ.U/V
forcing used by the solver.
"""
function discrete_barotropic_transport_budget(model, parameters)
    source_terms = barotropic_budget_outputs(model, parameters)
    stage_terms = map(zero_field_like, source_terms)

    template_u = stage_terms.total_u
    template_v = stage_terms.total_v

    previous_u = zero_field_like(template_u)
    previous_v = zero_field_like(template_v)

    derived = (transport_tendency_u = zero_field_like(template_u),
               transport_tendency_v = zero_field_like(template_v),
               solver_slow_u = zero_field_like(template_u),
               solver_slow_v = zero_field_like(template_v),
               component_rhs_u = zero_field_like(template_u),
               component_rhs_v = zero_field_like(template_v),
               slow_decomposition_residual_u = zero_field_like(template_u),
               slow_decomposition_residual_v = zero_field_like(template_v),
               split_explicit_pressure_u = zero_field_like(template_u),
               split_explicit_pressure_v = zero_field_like(template_v),
               closed_rhs_u = zero_field_like(template_u),
               closed_rhs_v = zero_field_like(template_v),
               closure_residual_u = zero_field_like(template_u),
               closure_residual_v = zero_field_like(template_v))

    outputs = merge(stage_terms, derived)
    budget = DiscreteTransportBudget(source_terms, outputs,
                                     previous_u, previous_v,
                                     0.0, false, false)
    return budget
end

"""Capture the individual tendencies used to construct the final RK-stage slow forcing."""
function capture_stage_two_budget!(model, budget::DiscreteTransportBudget)
    model.clock.stage == 2 || return nothing

    for name in keys(budget.source_terms)
        source = budget.source_terms[name]
        destination = budget.outputs[name]
        compute!(source)
        copy_interior_and_fill_halos!(destination, source)
    end

    budget.stage_two_captured = true
    return nothing
end

"""Update the exact discrete transport budget after a completed slow time step."""
function update_discrete_transport_budget!(model, budget::DiscreteTransportBudget)
    U = model.free_surface.barotropic_velocities.U
    V = model.free_surface.barotropic_velocities.V
    time = Float64(model.clock.time)
    outputs = budget.outputs

    if !budget.initialized
        copy_interior_and_fill_halos!(budget.previous_u, U)
        copy_interior_and_fill_halos!(budget.previous_v, V)
        budget.previous_time = time
        budget.initialized = true
        return nothing
    end

    Δt = time - budget.previous_time
    Δt > 0 || error("Non-positive elapsed time in discrete transport budget: Δt=$Δt")
    budget.stage_two_captured || error("Stage-2 transport-budget terms were not captured")

    interior(outputs.transport_tendency_u) .= (interior(U) .- interior(budget.previous_u)) ./ Δt
    interior(outputs.transport_tendency_v) .= (interior(V) .- interior(budget.previous_v)) ./ Δt

    # After the full step, these are the slow transport forcings used by the
    # final RK-stage split-explicit solve (including regular flux BCs).
    copy_interior_and_fill_halos!(outputs.solver_slow_u, model.timestepper.Gⁿ.U)
    copy_interior_and_fill_halos!(outputs.solver_slow_v, model.timestepper.Gⁿ.V)

    # The stage-2 total is the core Gⁿ forcing before regular FluxBoundaryCondition
    # contributions. Add +bottom -surface to reproduce the slow solver forcing.
    combine_fields!(outputs.component_rhs_u,
                    (outputs.total_u, 1),
                    (outputs.regular_bottom_u, 1),
                    (outputs.surface_u, -1))
    combine_fields!(outputs.component_rhs_v,
                    (outputs.total_v, 1),
                    (outputs.regular_bottom_v, 1),
                    (outputs.surface_v, -1))

    combine_fields!(outputs.slow_decomposition_residual_u,
                    (outputs.solver_slow_u, 1),
                    (outputs.component_rhs_u, -1))
    combine_fields!(outputs.slow_decomposition_residual_v,
                    (outputs.solver_slow_v, 1),
                    (outputs.component_rhs_v, -1))

    # This is the exact net external-mode pressure impulse per unit time,
    # including the fast-subcycle filter and time centering used by the solver.
    combine_fields!(outputs.split_explicit_pressure_u,
                    (outputs.transport_tendency_u, 1),
                    (outputs.solver_slow_u, -1))
    combine_fields!(outputs.split_explicit_pressure_v,
                    (outputs.transport_tendency_v, 1),
                    (outputs.solver_slow_v, -1))

    combine_fields!(outputs.closed_rhs_u,
                    (outputs.component_rhs_u, 1),
                    (outputs.slow_decomposition_residual_u, 1),
                    (outputs.split_explicit_pressure_u, 1))
    combine_fields!(outputs.closed_rhs_v,
                    (outputs.component_rhs_v, 1),
                    (outputs.slow_decomposition_residual_v, 1),
                    (outputs.split_explicit_pressure_v, 1))

    combine_fields!(outputs.closure_residual_u,
                    (outputs.transport_tendency_u, 1),
                    (outputs.closed_rhs_u, -1))
    combine_fields!(outputs.closure_residual_v,
                    (outputs.transport_tendency_v, 1),
                    (outputs.closed_rhs_v, -1))

    for field in values(derived_transport_budget_outputs(outputs))
        fill_halo_regions!(field)
    end

    copy_interior_and_fill_halos!(budget.previous_u, U)
    copy_interior_and_fill_halos!(budget.previous_v, V)
    budget.previous_time = time
    budget.stage_two_captured = false
    return nothing
end

Oceananigans.run_diagnostic!(diagnostic::DiscreteTransportBudgetDiagnostic, model) =
    update_discrete_transport_budget!(model, diagnostic.budget)

function derived_transport_budget_outputs(outputs)
    return (; transport_tendency_u = outputs.transport_tendency_u,
              transport_tendency_v = outputs.transport_tendency_v,
              solver_slow_u = outputs.solver_slow_u,
              solver_slow_v = outputs.solver_slow_v,
              component_rhs_u = outputs.component_rhs_u,
              component_rhs_v = outputs.component_rhs_v,
              slow_decomposition_residual_u = outputs.slow_decomposition_residual_u,
              slow_decomposition_residual_v = outputs.slow_decomposition_residual_v,
              split_explicit_pressure_u = outputs.split_explicit_pressure_u,
              split_explicit_pressure_v = outputs.split_explicit_pressure_v,
              closed_rhs_u = outputs.closed_rhs_u,
              closed_rhs_v = outputs.closed_rhs_v,
              closure_residual_u = outputs.closure_residual_u,
              closure_residual_v = outputs.closure_residual_v)
end

"""Barotropic transports and free surface saved at averaging-window endpoints."""
function barotropic_budget_state_outputs(model)
    U = model.free_surface.barotropic_velocities.U
    V = model.free_surface.barotropic_velocities.V
    η = model.free_surface.η
    return (; transport_u=U, transport_v=V, free_surface=η)
end
