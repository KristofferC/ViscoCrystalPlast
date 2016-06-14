module ViscoCrystalPlast

using ContMechTensors
using JuAFEM
using CALFEM
using Parameters
using ForwardDiff
using TimerOutputs
using NearestNeighbors
using GeometricalPredicates
using BlockArrays
using AffineTransforms

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


import ForwardDiff.GradientNumber

abstract QuadratureData

include("ComsolMeshReader.jl")
include("material_parameters.jl")

include("mesh.jl")
include("boundary_conditions.jl")
include("sparse_tools.jl")
include("mesh_transfer.jl")

include("mesh_readers/mesh_reader.jl")

include("utilities.jl")

include("primal/PrimalProblem.jl")

immutable PrimalProblem{T} <: AbstractProblem
    global_problem::PrimalGlobalProblem{T}
end


function PrimalProblem(nslips, fspace::JuAFEM.FunctionSpace)
    PrimalProblem(PrimalGlobalProblem(nslips, fspace))
end

include("primal/quadrature_data.jl")
include("primal/intf_primal.jl")
include("primal/global_problem.jl")



include("dual/DualProblem.jl")

immutable DualProblem{dim, T, N} <: AbstractProblem
    local_problem::DualLocalProblem{dim, T}
    global_problem::DualGlobalProblem{dim, T, N}
end


function DualProblem{dim}(nslips, fspace::JuAFEM.FunctionSpace{dim})
    DualProblem(DualLocalProblem(nslips, Dim{dim}), DualGlobalProblem(nslips, fspace))
end

include("dual/local_problem.jl")
include("dual/quadrature_data.jl")
include("dual/intf_dual.jl")
include("dual/global_problem.jl")

include("solve_problem.jl")

end # module
