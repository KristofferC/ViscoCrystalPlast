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
DEFAULT_PROJ_QUEUE[HEBBE] = ("SNIC2017-1-224", "hebbe")

modules = String[]
cluster == HEBBE && push!(modules, "intel")
cluster == GLENN && append!(modules, ["gcc", "mkl"])

extra = ""
if cluster == HEBBE
    extra = ". /apps/new_modules.sh"
end

if cluster == HEBBE
    julia_path = "/c3se/NOBACKUP/users/kricarl/julia/julia"
elseif cluster == GLENN
    julia_path = "/c3se/NOBACKUP/users/kricarl/julia_glenn/julia"
end

jobname = "1grain"
n_nodes = 1
n_cores = 2
if cluster == GLENN
    n_cores = 16
end
#proj_queue = DEFAULT_PROJ_QUEUE[cluster]
proj_queue = ("C3SE507-15-6", "mob")
array = (1,20)
time = "0-10:20:00"
julia_string = raw"""
include("experiment_with_code.jl")

n_realizations_per_job = 25

for bc in [ViscoCrystalPlast.Neumann, ViscoCrystalPlast.Dirichlet]
    for xi_bc = [:microfree, :microhard]
        for n in [1,]
            arr_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
            r = n_realizations_per_job * (arr_id - 1) + 1
            range = r : r + n_realizations_per_job - 1
            for id in range
              runit("meshes/n$n-id1.inp", bc, id; xi_bc_type = xi_bc)
            end
        end
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
