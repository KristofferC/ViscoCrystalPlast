const RUNNING_SLURM = haskey(ENV, "SLURM_JOBID")
const RUNNING_SLURM_ARRAY = haskey(ENV, "SLURM_ARRAY_TASK_ID")


rcl = 0.55
n_realizatons = 1
dim = 3
if !haskey(ENV, "NEPER_PATH")
    neper_path = "/home/kristoffer/neper/build/neper"
else
    neper_path = joinpath(ENV["HOME"], "neper-3.0.2/build/neper")
end

n_realizations_per_job = 5
n_polys = [50]

for n in n_polys
    if RUNNING_SLURM_ARRAY
        cd(ENV["TMPDIR"])
        arr_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
        r = n_realizations_per_job * (arr_id - 1) + 1
        range = r : r + n_realizations_per_job - 1
        # range = 1:n_realizatons
        println("Running slurm with id: $range")
    else
        range = 1:n_realizatons
    end
    for r in range
        filename = "n$n-id$r"
        run(`$neper_path -T -n $n -id $r -dim $dim -morpho "diameq:dirac(1),sphericity:lognormal(0.145,0.03,1-x)"`)
        run(`$neper_path -M n$n-id$r.tess -dim all -rcl $rcl -format inp`)
        mv(joinpath(pwd(), filename * ".tess"), joinpath(@__DIR__, "meshes", filename * ".tess"); remove_destination=true)
        mv(joinpath(pwd(), filename * ".inp" ), joinpath(@__DIR__, "meshes", filename * ".inp"); remove_destination=true)
    end
end
