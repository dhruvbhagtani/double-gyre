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

### Running and restarting with Slurm

Every submission must explicitly state whether it starts a fresh simulation or
resumes from a checkpoint. `STOP_YEARS` is the absolute final model time, not
the number of years to add during the submitted job.

Start a fresh one-year run with:

```bash
sbatch --export=ALL,RUN_MODE=fresh,STOP_YEARS=1 run_double_gyre_gpu.sh
```

Resume the saved one-year state and integrate through year two with:

```bash
sbatch --export=ALL,RUN_MODE=resume,STOP_YEARS=2,CHECKPOINT=checkpoints/double_gyre_iteration17520.jld2 run_double_gyre_gpu.sh
```

`RUN_MODE` accepts only `fresh` or `resume`. Resume runs require `CHECKPOINT`
to name a specific checkpoint file; fresh runs require it to be unset. The
legacy `PICKUP` variable is no longer supported. A resume also fails early if
the checkpoint does not exist or its saved model time already equals or exceeds
the requested `STOP_YEARS`.
