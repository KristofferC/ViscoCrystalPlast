#include("../src/ViscoCrystalPlast.jl")

using ViscoCrystalPlast
using JuAFEM
using ForwardDiff
using ContMechTensors
using TimerOutputs
using NLsolve

import ViscoCrystalPlast: GeometryMesh, DofHandler, DirichletBoundaryConditions

import ViscoCrystalPlast: dofs_element, element_coordinates!
import ViscoCrystalPlast: assemble!, create_sparsity_pattern, read_mphtxt

import ViscoCrystalPlast: add_field!, close!, element_set, element_coordinates,
                            add_dirichletbc!, ndim, ndofs, nelements, nnodes, free_dofs, update_dirichletbcs!, apply!,
                                create_lookup,


function element_internal_forces{dim, T, V <: Vec}(primary_field::Vector{T},
                                             x::Vector{V}, fev::FEValues{dim}, C)
    n_basefuncs = n_basefunctions(get_functionspace(fev))
    u_vec = reinterpret(Vec{dim, T}, primary_field, (n_basefuncs,))
    reinit!(fev, x)

    fe_u = [zero(Vec{dim, T}) for i in 1:n_basefuncs]
    Ke = zeros(n_basefuncs * dim, n_basefuncs * dim)

    @inbounds for q_point in 1:length(points(get_quadrule(fev)))
        ε = function_vector_symmetric_gradient(fev, q_point, u_vec)
        σ = C ⊡ ε
        for i in 1:n_basefuncs
            fe_u[i] += σ ⋅ shape_gradient(fev, q_point, i) * detJdV(fev, q_point)
            for j in 1:n_basefuncs
                Kee = dotdot(shape_gradient(fev, q_point, i), C, shape_gradient(fev, q_point, j))  * detJdV(fev, q_point)
                for I in 1:dim, J in 1:dim
                    Ke[(i-1) * dim + I, (j-1) * dim + J] += Kee[I, J]
                end
            end
        end
    end

    return Ke, reinterpret(T, fe_u, (dim * n_basefuncs,))
end




function get_stiffness{dim}(E, ν, ::Type{Dim{dim}})
    λ = E*ν / ((1+ν) * (1 - 2ν))
    μ = E / (2(1+ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    f(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l) * δ(j,k))
    Ee = SymmetricTensor{4, dim}(f)
end


function assemble!{dim}(K::SparseMatrixCSC, primary_field::Vector, fe_values::FEValues{dim}, mesh::GeometryMesh, dh::DofHandler,
                  bcs::DirichletBoundaryConditions, C, lookup)

    fill!(K.nzval, 0.0)
    n_basefuncs = n_basefunctions(get_functionspace(fe_values))

    dofs_per_element = length(dofs_element(dh, 1))
    f_int = zeros(ndofs(dh))
    K_element = zeros(dofs_per_element, dofs_per_element)

    #local element_coords
    #local primary_element_field
    #fe(field) = element_internal_forces!(field, element_coords, fe_values, C)
    #Ke! = ForwardDiff.jacobian(fe, ForwardDiff.AllResults, mutates = true, chunk_size = 6)

    a = start_assemble()
    for element_id in 1:nelements(mesh)
        edof = dofs_element(dh, element_id)
        element_coords = element_coordinates(mesh, element_id)
        primary_element_field = primary_field[edof]

        Ke, fe = element_internal_forces(primary_element_field, element_coords, fe_values, C)

        assemble!(f_int, fe , edof)
        assemble!(K, Ke, element_id, lookup)
    end
    free = free_dofs(bcs)
    return K[free, free], f_int[free]
end


function solve_problem{dim}(mesh, dh, bcs, fe_values::FEValues{dim}, C)
    reset_timer!(to)
    free = free_dofs(bcs)

    K = create_sparsity_pattern(dh)
    ∆u = zeros(length(free))
    primary_field = zeros(ndofs(dh))
    test_field = copy(primary_field)
    pvd = paraview_collection("vtks/shear_dual")
    lookup = create_lookup(K, dh)

    step = 0
    for t in 1.0:20.0
        update_dirichletbcs!(bcs, t)
        apply!(primary_field, bcs)
        copy!(test_field, primary_field)

        iter = 1
        n_iters = 10

        @time while true
            test_field[free] = primary_field[free] + ∆u
            K_condensed, f = @timeit to "Ke" begin
                 assemble!(K, test_field, fe_values, mesh, dh, bcs, C, lookup)
            end

            println(norm(f, Inf))
            if norm(f, Inf) / norm(C)  <= 1e-9
                println("converged")
                break
            end

            @timeit to "linear solve" begin
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
            vtkoutput(pvd, t, step+=1, mesh, bcs, primary_field, dim)
        end
    end
    vtk_save(pvd)
    print(to)
end


function vtkoutput(pvd, time, step, mesh, dbc, primary_field, dim)

    vtkfile = vtk_grid(mesh, "vtks/box_$step")
    vtk_point_data(vtkfile, dbc)

    primary_field_mat = reshape(primary_field, (dim, nnodes(mesh)))
    if dim == 2
        primary_field_mat = [primary_field_mat; zeros(nnodes)']
    end

    vtk_point_data(vtkfile, primary_field_mat, "displacement")
    collection_add_timestep(pvd, vtkfile, time)
end


function startit()
    #mesh = create_mesh("../test/test_mesh.mphtxt")
    mesh = GeometryMesh(read_mphtxt("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/3d_cube.mphtxt"))
    dh = DofHandler(mesh)
    add_field!(dh, :u, 3)
    close!(dh)
    dbcs = DirichletBoundaryConditions(dh)
    add_dirichletbc!(dbcs, :u, element_set(mesh, "boundary nodes"), (x,t) -> t * x * 0.01, [1, 2, 3])
    close!(dbcs)

    E = 200e9
    ν = 0.3
    C = get_stiffness(E, ν, Dim{3})

    function_space = JuAFEM.Lagrange{3, RefTetrahedron, 1}()
    q_rule = QuadratureRule(Dim{3}, RefTetrahedron(), 2)
    fe_values = FEValues(Float64, q_rule, function_space)


    solve_problem(mesh, dh, dbcs, fe_values, C)
end


const to = TimerOutput();
startit()
