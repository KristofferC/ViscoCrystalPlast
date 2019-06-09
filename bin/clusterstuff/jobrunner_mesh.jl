include("clustermanager.jl")

@enum Cluster HEBBE GLENN


if startswith(ENV["HOSTNAME"], "glenn")
    cluster = GLENN
elseif startswith(ENV["HOSTNAME"], "hebbe")
    cluster = HEBBE
else
    error("Unknown cluster")
end

const DEFAULT_PROJ_QUEUE = Dict{Cluster, Tuple{String,String}}()
DEFAULT_PROJ_QUEUE[GLENN] = ("C3SE2017-1-8", "glenn")
#DEFAULT_PROJ_QUEUE[HEBBE] = ("C3SE2017-1-8", "hebbe")
DEFAULT_PROJ_QUEUE[HEBBE] = ("SNIC2017-1-224", "hebbe")

modules = String[]
cluster == HEBBE && push!(modules, "intel")
cluster == GLENN && append!(modules, ["gcc", "mkl"])
append!(modules, ["GSL", "gmsh"])

extra = ""
if cluster == HEBBE
    extra = ". /apps/new_modules.sh"
end

if cluster == HEBBE
    julia_path = "/c3se/NOBACKUP/users/kricarl/julia/julia"
elseif cluster == GLENN
    julia_path = "/c3se/NOBACKUP/users/kricarl/julia_glenn/julia"
end

jobname = "mesh100poly"
n_nodes = 1
n_cores = 1
if cluster == GLENN
    n_cores = 16
end
proj_queue = DEFAULT_PROJ_QUEUE[cluster]
#proj_queue = ("C3SE507-15-6", "mob")
array = (1,5)
time = "0-03:00:00"
julia_string = raw"""
const RUNNING_SLURM = haskey(ENV, "SLURM_JOBID")
const RUNNING_SLURM_ARRAY = haskey(ENV, "SLURM_ARRAY_TASK_ID")


rcl = 0.65
n_realizatons = 1
dim = 3
if !RUNNING_SLURM
    neper_path = "/home/kristoffer/neper/build/neper"
else
    neper_path = joinpath(ENV["HOME"], "neper-3.0.2/build/neper")
end

n_realizations_per_job = 4
n_polys = [100]

for n in n_polys
    if RUNNING_SLURM_ARRAY
        cd(ENV["TMPDIR"])
        neper_dir = ENV["TMPDIR"]
        target_dir = ENV["SLURM_SUBMIT_DIR"]
        arr_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
        r = n_realizations_per_job * (arr_id - 1) + 1
        range = r : r + n_realizations_per_job - 1
        # range = 1:n_realizatons
        println("Running slurm with id: $range and neper_dir $neper_dir")
    else
        neper_dir = @__DIR__
        target_dir = @__DIR__
        range = 1:n_realizatons
    end
    for r in range
        filename = "n$n-id$r"
        run(`$neper_path -T -n $n -id $r -dim $dim -morpho "diameq:dirac(1),sphericity:lognormal(0.145,0.03,1-x)"`)
        run(`$neper_path -M n$n-id$r.tess -dim all -rcl $rcl -format inp`)
        mv(joinpath(neper_dir, filename * ".tess"), joinpath(target_dir, "meshes", filename * ".tess"); remove_destination=true)
        mv(joinpath(neper_dir, filename * ".inp" ), joinpath(target_dir, "meshes", filename * ".inp"); remove_destination=true)
    end
end
"""

queue_job(proj_queue,
          jobname = jobname,
          n_nodes = n_nodes,
          n_cores = n_cores,
          array = array,
          modules = modules,
          julia_string = julia_string,
          time = time,
          extra = extra,
          julia_path = julia_path)
