module ViscoCrystalPlast

using ContMechTensors
using JuAFEM
using CALFEM
using Parameters
using ForwardDiff
using TimerOutputs

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

abstract AbstractProblem

immutable DualProblem <: AbstractProblem end
immutable PrimalProblem <: AbstractProblem end

import ForwardDiff.GradientNumber

include("ComsolMeshReader.jl")
include("material_parameters.jl")
include("newton_problem.jl")

include("mesh.jl")
include("boundary_conditions.jl")
include("sparse_tools.jl")
include("quadrature_data.jl")

include("utilities.jl")
include("solve_problem.jl")
include("primal/solve_primal.jl")
include("dual/solve_dual.jl")



end # module