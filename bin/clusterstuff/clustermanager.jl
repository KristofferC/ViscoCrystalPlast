function queue_job(project_queue::Tuple{String, String}; jobname::String = "TestJob", n_nodes::Int = 1, n_cores::Int = 1, time::String="0-00:05:00",
                    modules::Vector{String} = String[], extra = "", julia_path = "julia",
                   julia_string::String = "println(\"Hello world\")", array = nothing, debug = false)
   project, queue = project_queue
   f_path, f = mktemp()
   println(f, "#!/usr/bin/env bash")
   println(f, "#SBATCH -A $project")
   println(f, "#SBATCH -p $queue")
   println(f, "#SBATCH -J $jobname")
   println(f, "#SBATCH -N $n_nodes")
   println(f, "#SBATCH -n $n_cores")
   println(f, "#SBATCH -t $time")

   println(f, extra)

   for m in modules
       println(f, "module load $m")
   end
   println()
   cmd = ""
   if array != nothing
       @assert typeof(array) == Int || typeof(array) == Tuple{Int, Int}
       cmd *= "--array="
       if isa(array, Int)
           cmd *= string(array)
       else
           cmd *= string(array[1], "-", array[2])
       end
   end

   println(f, "$julia_path -e '")
   print(f, julia_string)
   println(f, "'")
   close(f)

   if debug
       println("Running: $cmd $f_path with file: \n ", readstring(f_path))
       @async run(`$cmd $f_path`)
   else
       println("Running: sbatch $cmd $f_path with file: \n ", readstring(f_path))
       @async run(`sbatch $cmd $f_path`)
   end
end
