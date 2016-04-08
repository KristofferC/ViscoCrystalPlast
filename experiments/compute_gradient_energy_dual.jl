using ViscoCrystalPlast
using JuAFEM
using ForwardDiff
using ContMechTensors
using TimerOutputs


import ViscoCrystalPlast: GeometryMesh, Dofs, DirichletBoundaryConditions, CrystPlastMP
import ViscoCrystalPlast: create_mesh, add_dofs, dofs_element, element_coordinates

function boundary_f(field::Symbol, x, t::Float64)
    if field == :u
        return 0.01 * x[2] * t
    elseif field == :v
        return 0.01 * x[2] * t
    elseif field == :w
        return 0.0
    else
        error("unhandled field")
    end
end

function startit()
    reset_timer!()
    mesh = ViscoCrystalPlast.create_mesh("../test/test_mesh.mphtxt")
   # mesh = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/3d_cube.mphtxt")
 #  mesh = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/test_mesh.mphtxt")
    mp = setup_material(Dim{2})

    dofs = ViscoCrystalPlast.add_dofs(mesh, [:u, :v, :ξ⟂1, :ξ⟂2])

    bcs = ViscoCrystalPlast.DirichletBoundaryConditions(dofs, mesh.boundary_nodes, [:u, :v])

    function_space = JuAFEM.Lagrange{2, RefTetrahedron, 1}()
    q_rule = QuadratureRule(Dim{2}, RefTetrahedron(), 1)
    fe_values = FEValues(Float64, q_rule, function_space)

    times = linspace(0.0, 1.0, 4)
    ViscoCrystalPlast.solve_problem(ViscoCrystalPlast.DualProblem(), mesh, dofs, bcs, fe_values, mp, times, boundary_f, (i)->i)
    print_timer()
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


