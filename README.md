# double-gyre
Double gyre simulations with Oceananigans


### Instructions

First [install Julia](https://julialang.org/downloads/); suggested version 1.10. See [juliaup](https://github.com/JuliaLang/juliaup) README for how to install 1.10 and make that version the default.

Then clone this repository

```bash
git clone git@github.com:dhruvbhagtani/double-gyre.git
```

Open Julia from within the local directory of the repo via:

```bash
julia --project
```

The first time, you need to install any dependencies:

```julia
julia> using Pkg; Pkg.instantiate()
```

Now you are ready to run the main script!

For instance,

```julia
julia> include("double-gyre.jl")
```

### Discrete barotropic-vorticity budget

The GPU submission script runs the model with an online, discrete
depth-integrated transport budget:

```bash
sbatch run_double_gyre_gpu.sh
```

`double_gyre_barotropic_budget.jld2` contains the stage-used physical slow
terms, the exact step-to-step transport tendency, the slow forcing actually
used by the split-explicit solver, and the effective split-explicit pressure
contribution. The two principal checks are

```text
solver_slow = component_rhs + slow_decomposition_residual
transport_tendency = solver_slow + split_explicit_pressure
```

and `closure_residual = transport_tendency - closed_rhs` should be near roundoff.

After the run, convert every vector term to its native spherical C-grid curl.
The default output is a time-weighted yearly mean:

```bash
julia --project=. --startup-file=no convert_barotropic_vorticity_budget.jl --force
```

This writes `double_gyre_barotropic_vorticity_budget.jld2`, with one record at
the end of each model year. To retain the source monthly records instead, use:

```bash
julia --project=. --startup-file=no convert_barotropic_vorticity_budget.jl \
    --time-resolution=monthly --force
```

Monthly output is written to
`double_gyre_barotropic_vorticity_budget_monthly.jld2`. Its day-zero tendency
is initialization output and should not be included in averages.

When the state file is present, the output also includes `beta_V`,
`eta_tendency_term`, and `bottom_pressure_torque`, matching the barotropic
vorticity budget of Khatri et al. (2024, JAMES, Eq. 3/A17):

```text
beta_V = bottom_pressure_torque + wind_stress - bottom_drag + advection + closure
         - eta_tendency_term - total
```

`bottom_pressure_torque` (J(pb,H)/ρo) is diagnosed from the Eq. B4 combination
`coriolis + pressure + beta_V - eta_tendency_term` rather than a raw curl,
since the paper shows the raw Coriolis and pressure curls are individually
dominated by C-grid numerical noise. This assumes zero surface mass flux
(Qm = 0), which holds for this closed-basin, no-flux configuration.

