using ViscoCrystalPlast
using JuAFEM
using ForwardDiff
using ContMechTensors
using TimerOutputs


import ViscoCrystalPlast: GeometryMesh, Dofs, DirichletBoundaryConditions, CrystPlastMP
import ViscoCrystalPlast: create_mesh, add_dofs, dofs_element, element_coordinates


function startit()
    #mesh = ViscoCrystalPlast.create_mesh("../test/test_mesh.mphtxt")
    mesh = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/test_mesh.mphtxt")
    mp = setup_material(Dim{2})

    dofs = ViscoCrystalPlast.add_dofs(mesh, [:u, :v, :γ1, :γ2])

    bcs = ViscoCrystalPlast.DirichletBoundaryConditions(dofs, mesh.boundary_nodes, [:u, :v, :γ1, :γ2])

    function_space = JuAFEM.Lagrange{2, RefTetrahedron, 1}()
    q_rule = QuadratureRule(Dim{2}, RefTetrahedron(), 1)
    fe_values = FEValues(Float64, q_rule, function_space)


    ViscoCrystalPlast.solve_primal_problem(mesh, dofs, bcs, fe_values, mp, boundary_f)
end


function boundary_f(field::Symbol, x, t::Float64)
    if field == :u
        return 0.01 * x[1] * t
    elseif field == :v
        return 0.01 * x[2] * t
    elseif field == :w
        return 0.0
    elseif startswith(string(field), "γ")
        return 0.0
    else
        error("unhandled field")
    end
end

function setup_material{dim}(::Type{Dim{dim}})
    E = 200000.0
    ν = 0.3
    n = 2.0
    lα = 0.5
    H⟂ = 0.1E
    Ho = 0.1E
    C = 1.0e3
    tstar = 1000.0
    angles = [20.0, 40.0]
    mp = ViscoCrystalPlast.CrystPlastMP(Dim{dim}, E, ν, n, H⟂, Ho, lα, tstar, C, angles)
    return mp
end


const to = TimerOutput();
startit()


