module ViscoCrystalPlast

using ContMechTensors
using JuAFEM
using CALFEM
using Parameters
using ForwardDiff

const DEBUG = true

if DEBUG
    @eval begin
        macro dbg_assert(ex)
            return quote
                @assert($(esc(ex)))
            end
        end
    end
else
     @eval begin
        macro dbg_assert(ex)
            return quote
                $(esc(ex))
            end
        end
    end
end

include("ComsolMeshReader.jl")
include("material_parameters.jl")

include("mesh.jl")
include("boundary_conditions.jl")
include("sparse_tools.jl")


include("utilities.jl")
include("primal/solve_primal.jl")



end # module