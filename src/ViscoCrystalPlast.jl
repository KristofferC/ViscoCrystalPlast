module ViscoCrystalPlast

using ContMechTensors
using JuAFEM
using CALFEM
using Parameters
using ForwardDiff
using TimerOutputs
using NearestNeighbors
using GeometricalPredicates

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

abstract QuadratureData

include("ComsolMeshReader.jl")
include("material_parameters.jl")
#include("newton_problem.jl")

include("mesh.jl")
include("boundary_conditions.jl")
include("sparse_tools.jl")
include("mesh_transfer.jl")

include("utilities.jl")
include("solve_problem.jl")


include("primal/quadrature_data.jl")
include("primal/intf_primal.jl")

include("dual/quadrature_data.jl")
include("dual/intf_dual.jl")




end # module
