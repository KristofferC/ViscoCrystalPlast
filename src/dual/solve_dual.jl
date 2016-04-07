#include("../src/ViscoCrystalPlast.jl")

using ViscoCrystalPlast
using JuAFEM
using ForwardDiff
using ContMechTensors
using TimerOutputsa

import ViscoCrystalPlast: GeometryMesh, Dofs, DirichletBoundaryConditions

import ViscoCrystalPlast: create_mesh, add_dofs, dofs_element, element_coordinates!
import ViscoCrystalPlast: assemble!, create_sparsity_pattern

# f(x) -> v
function update_bcs!(mesh::GeometryMesh, dofs::Dofs, bc::DirichletBoundaryConditions, time::Float64, f)
    dofs_per_node = length(dofs.dof_types)
    for i in eachindex(bc.dof_ids)
        dof_type = bc.dof_types[i]
        dof_id = bc.dof_ids[i]
        node = div(dof_id + dofs_per_node -1, dofs_per_node)
        ViscoCrystalPlast.set_value(bc, f(dof_type, mesh.coords[:, node], time), i)
    end
end

function boundary_f(field::Symbol, x, t::Float64)
    if field == :u
        print(x[2])
        return 0.02 * x[2] * t
    elseif field == :v
        return 0.0
    elseif field == :w
        return 0.0
    else
        error("unhandled field")
    end
end


function element_internal_forces!{dim, T, Q}(primary_field::Vector{T},
                                             x::AbstractArray{Q}, fev::FEValues{dim}, C)
    n_basefuncs = n_basefunctions(get_functionspace(fev))
    x_vec = reinterpret(Vec{dim, Q}, x, (n_basefuncs,))
    u_vec = reinterpret(Vec{dim, T}, primary_field, (n_basefuncs,))
    reinit!(fev, x_vec)

    fe_u = [zero(Vec{dim, T}) for i in 1:n_basefuncs]

    for q_point in 1:length(points(get_quadrule(fev)))
        ε = function_vector_symmetric_gradient(fev, q_point, u_vec)
        σ = C * ε
        for i in 1:n_basefuncs
            fe_u[i] += σ ⋅ shape_gradient(fev, q_point, i) * detJdV(fev, q_point)
        end
    end

    return reinterpret(T, fe_u, (dim * n_basefuncs,))
end



function get_stiffness{dim}(E, ν, ::Type{Dim{dim}})
    λ = E*ν / ((1+ν) * (1 - 2ν))::Float64
    μ = E / (2(1+ν))::Float64
    δ(i,j) = i == j ? 1.0 : 0.0
    f(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l) * δ(j,k))
    Ee = SymmetricTensor{4, dim}(f)
end



function assemble!{dim}(K::SparseMatrixCSC, primary_field::Vector, fe_values::FEValues{dim}, mesh::GeometryMesh, dofs::Dofs,
                  bcs::DirichletBoundaryConditions, C)

    @assert length(primary_field) == length(dofs.dof_ids)
    fill!(K.nzval, 0.0)
    n_basefuncs = n_basefunctions(get_functionspace(fe_values))
    dofs_per_node = length(dofs.dof_types)
    dofs_per_element = n_basefuncs * dofs_per_node

    f_int = zeros(length(dofs.dof_ids))
    K_element = zeros(dofs_per_element, dofs_per_element)

    local e_coordinates
    local primary_element_field
    fe(field) = element_internal_forces!(field, e_coordinates, fe_values, C)
    Ke! = ForwardDiff.jacobian(fe, ForwardDiff.AllResults, mutates = true, chunk_size = 6)


    e_coordinates = zeros(dim, n_basefuncs)
    a = start_assemble()
    for element_id in 1:size(mesh.topology, 2)
        edof = dofs_element(mesh, dofs, element_id)
        element_coordinates!(e_coordinates , mesh, element_id)
        primary_element_field = primary_field[edof]

        Ke, allresults = Ke!(K_element, primary_element_field)
        f_int[edof] +=  ForwardDiff.value(allresults)

        #assemble!(f_int, ForwardDiff.value(allresults), edof)
        assemble!(K, Ke, edof)
    end
  #  print(K)
    free = setdiff(dofs.dof_ids, bcs.dof_ids)
    return K[free, free], f_int[free]
end


function solve_problem{dim}(mesh, dofs, bcs, fe_values::FEValues{dim}, C)
    free = setdiff(dofs.dof_ids, bcs.dof_ids)


    K = create_sparsity_pattern(mesh, dofs)
    ∆u = zeros(length(free))
    primary_field = zeros(length(dofs.dof_ids))
    test_field = copy(primary_field)
    pvd = paraview_collection("vtks/shear_dual")

    step = 0
    for t in 1.0:5.0
        update_bcs!(mesh, dofs, bcs, t, boundary_f)
        primary_field[bcs.dof_ids] = bcs.values
        copy!(test_field, primary_field)

        iter = 1
        n_iters = 10
        while iter == 1 || norm(f, Inf) / norm(C)  >= 1e-9
            test_field[free] = primary_field[free] + ∆u
            K_condensed, f = @timeit to "Ke" begin
                assemble!(K, test_field, fe_values, mesh, dofs, bcs, C)
            end
            print(K_condensed)

            @timeit to "solve" begin
                 ∆u -=  K_condensed \ f
                 #∆u -=  cholfact(Symmetric(K_condensed, :U)) \ f
            end

            iter +=1
            if n_iters == iter
                error("too many iterations without convergence")
            end
        end
        copy!(primary_field, test_field)
        @timeit to "vtk" begin
            vtkoutput(pvd, t, step+=1, mesh, primary_field, dim)
        end
    end
    vtk_save(pvd)
    print(to)
end


function vtkoutput(pvd, time, step, mesh, primary_field, dim)

    vtkfile = vtk_grid(mesh.topology, mesh.coords, "vtks/box_$step")
    nnodes = size(mesh.coords, 2)

    primary_field_mat = reshape(primary_field, (dim, nnodes))
    if dim == 2
        primary_field_mat = [primary_field_mat; zeros(nnodes)']
    end

    vtk_point_data(vtkfile, primary_field_mat, "displacement")
    collection_add_timestep(pvd, vtkfile, time)
end


function startit()
    #mesh = create_mesh("../test/test_mesh.mphtxt")
    mesh = create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/3d_cube.mphtxt")

    dofs = add_dofs(mesh, [:u, :v, :w])
    bcs = DirichletBoundaryConditions(dofs, mesh.boundary_nodes, [:u, :v, :w])
    E = 200e9
    ν = 0.3
    C = get_stiffness(E, ν, Dim{3})

    function_space = JuAFEM.Lagrange{3, RefTetrahedron, 1}()
    q_rule = QuadratureRule(Dim{3}, RefTetrahedron(), 2)
    fe_values = FEValues(Float64, q_rule, function_space)


    solve_problem(mesh, dofs, bcs, fe_values, C)
end


const to = TimerOutput();
startit()
