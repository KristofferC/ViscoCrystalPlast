module ViscoCrystalPlast

using Tensors
using JuAFEM
using ProgressMeter
using Parameters
using ForwardDiff
using TimerOutputs
using FileIO
using BlockArrays
import JuAFEM.vtk_point_data
using MUMPS
using Pardiso
using Compat

@compat abstract type AbstractProblem end

DEBUG = true

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

@compat abstract type QuadratureData end

include("material_parameters.jl")

include("mesh_reader.jl")
include("mesh.jl")
include("sparse_tools.jl")
include("mesh_transfer.jl")
include("mesh_utils.jl")
include("utilities.jl")

include("primal/PrimalProblem.jl")

immutable PrimalProblem{T} <: AbstractProblem
    global_problem::PrimalGlobalProblem{T}
end


function PrimalProblem{dim}(nslips, fev_u::CellVectorValues{dim}, fev_γ::CellScalarValues{dim})
    PrimalProblem(PrimalGlobalProblem(nslips, fev_u, fev_γ))
end

include("primal/quadrature_data.jl")
include("primal/global_problem.jl")
include("dual/DualProblem.jl")

immutable DualProblem{dim, T, N} <: AbstractProblem
    local_problem::DualLocalProblem{dim, T}
    global_problem::DualGlobalProblem{dim, T, N}
end


function DualProblem{dim}(nslips, fev_u::CellVectorValues{dim}, fev_ξ::CellScalarValues{dim})
    DualProblem(DualLocalProblem(nslips, Dim{dim}), DualGlobalProblem(nslips, fev_u, fev_ξ))
end

include("dual/local_problem.jl")
include("dual/quadrature_data.jl")
include("dual/global_problem.jl")

include("solve_problem.jl")

end # module
