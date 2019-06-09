module Homogenization

using JuAFEM
using Tensors
using TimerOutputs
using UnicodePlots
using ForwardDiff
const to = TimerOutput();

DIM = 3
GEOSHAPE = DIM == 2 ? Triangle : Tetrahedron
REFSHAPE = RefTetrahedron
QUADRATURE_ORDER = 1


function compute_grid(n, f)
    grid = generate_grid(GEOSHAPE, ntuple(i -> n, DIM));
    # Extract the left boundary
    addnodeset!(grid, "clamped", f);
end

# Interpolations and values
function create_interpolations()
    interpolation_space = Lagrange{DIM, REFSHAPE, 1}()
    quadrature_rule = QuadratureRule{DIM, REFSHAPE}(QUADRATURE_ORDER)
    face_quadrature_rule = QuadratureRule{DIM-1, REFSHAPE}(QUADRATURE_ORDER)
    cellvalues = CellVectorValues(quadrature_rule, interpolation_space);
    facevalues = FaceVectorValues(face_quadrature_rule , interpolation_space);
    return cellvalues, facevalues
end

function create_dofhandler(grid)
    # DofHandler
    dh = DofHandler(grid)
    push!(dh, :u, DIM) # Add a displacement field
    close!(dh)
    return dh
end

function create_boundaryconditions(grid, dh, f)
    # Boundaryconditions
    dbc = DirichletBoundaryConditions(dh)
    # Add a homogenoush boundary condition on the "clamped" edge
    add!(dbc, :u, getnodeset(grid, "clamped"), f, collect(1:DIM))
    close!(dbc)
    t = 0.0
    update!(dbc, t)
    return dbc
end

# Create the stiffness tensor
function create_stiffness(Emod = 200e9, ν = 0.3)
    λ = Emod*ν / ((1+ν) * (1 - 2ν))
    μ = Emod / (2(1+ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    g(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
    E = SymmetricTensor{4, DIM}(g)
    return E
end



function doassemble!{dim}(cellvalues::CellVectorValues{dim}, facevalues::FaceValues, ɛ_bar, u, σ_bar,
                           K::SparseMatrixCSC, grid::Grid, dh::DofHandler, E::SymmetricTensor{4, dim})

    f = zeros(ndofs(dh))
    residual_u = zeros(f)
    C = zeros(ndofs(dh), 9)
    assembler = start_assemble(K, f)

    n_basefuncs = getnbasefunctions(cellvalues)
    global_dofs = zeros(Int, ndofs_per_cell(dh))

    Ke = zeros(n_basefuncs, n_basefuncs) # Local stiffness mastrix
    Ce = zeros(n_basefuncs, 9)
    Ω = 8

    hess_res = DiffBase.HessianResult(rand(n_basefuncs + 9))
    hess_cfg = ForwardDiff.HessianConfig{3}(hess_res, rand(n_basefuncs + 9))

    i = 0
    @inbounds for cell in CellIterator(dh)
        celldofs!(global_dofs, cell)
        u_element = u[global_dofs]
        uσ_bar = [u_element; σ_bar]

        ff = x -> compute_element_potential!(cellvalues, cell,
                                        x, ɛ_bar, E)

        ForwardDiff.hessian!(hess_res, ff, uσ_bar, hess_cfg)
        K = DiffBase.hessian(hess_res)
        res = DiffBase.gradient(hess_res)
        res_u = res[1:n_basefuncs]
        res_c = res[n_basefuncs+1:end]
        K_uu = K[1:n_basefuncs, 1:n_basefuncs]
        K_uC = K[1:n_basefuncs, n_basefuncs+1:end]

        residual_ue, residual_C, Kuu, KuC = compute_element_residual_and_stiffness!(cellvalues, cell,
                                                        uσ_bar, ɛ_bar, E)

        for (i, dof) in enumerate(global_dofs)
            C[dof, :] += K_uC[i, :]
        end
        assemble!(residual_u,  res_u, global_dofs)
        assemble!(assembler, K_uu, global_dofs)
    end
    return f, C, residual_u
end

function compute_element_potential!{dim, T}(cellvalues::CellVectorValues{dim}, cellcoords,
                                            uσ_bar::Vector{T}, ɛ_bar, E::SymmetricTensor{4, dim})
    n_basefuncs = getnbasefunctions(cellvalues)
    u, σ_bar = uσ_bar[1:n_basefuncs], uσ_bar[n_basefuncs+1:end]
    uuv = reinterpret(Vec{dim, T}, u, (4,))
    σv = fromvoigt(Tensor{2, dim}, σ_bar)
    Ω = 8
    Π = 0.0
    reinit!(cellvalues, cellcoords)
    for q_point in 1:getnquadpoints(cellvalues)
        ∇u_qp = function_gradient(cellvalues, q_point, uuv)
        ɛ_qp = symmetric(∇u_qp)
        dΩ = getdetJdV(cellvalues, q_point)
        Π += 1/Ω * (1/2 * ɛ_qp ⊡ E ⊡ ɛ_qp - σv ⊡ (∇u_qp - ɛ_bar)) * dΩ
    end
    return Π
end


function compute_element_residual_and_stiffness!{dim, T}(cellvalues::CellVectorValues{dim}, cellcoords,
                                                        uσ_bar::Vector{T}, ɛ_bar, E::SymmetricTensor{4, dim})
    n_basefuncs = getnbasefunctions(cellvalues)
    residual_ue = zeros(n_basefuncs)
    residual_C = zero(Tensor{2,3})
    Kuu = zeros(n_basefuncs, n_basefuncs)
    KuC = zeros(n_basefuncs, length(residual_C))
    u, σ_bar = uσ_bar[1:n_basefuncs], uσ_bar[n_basefuncs+1:end]
    uuv = reinterpret(Vec{dim, T}, u, (4,))
    σv = fromvoigt(Tensor{2, dim}, σ_bar)
    Ω = 8

    reinit!(cellvalues, cellcoords)
    for q_point in 1:getnquadpoints(cellvalues)
        ∇u_qp = function_gradient(cellvalues, q_point, uuv)
        ɛ_qp = symmetric(∇u_qp)
        dΩ = getdetJdV(cellvalues, q_point)
        residual_C += -1/Ω * (∇u_qp - ɛ_bar) * dΩ
        for i in 1:n_basefuncs
            δ∇ui = shape_gradient(cellvalues, q_point, i)
            δɛi = symmetric(shape_gradient(cellvalues, q_point, i))
            residual_ue[i] +=  1/Ω * (δɛi ⊡ E ⊡ ɛ_qp - σv ⊡ δ∇ui) * dΩ
            KuC[i, :] += -1/Ω * tovoigt(δ∇ui) * dΩ
            for j in 1:n_basefuncs
                δɛj = symmetric(shape_gradient(cellvalues, q_point, j))
                Kuu[i, j] += 1/Ω * (δɛi ⊡ E ⊡ δɛj) * dΩ
            end
        end
    end
    return residual_ue, residual_C, Kuu, KuC
end

function check{dim}(cellvalues::CellVectorValues{dim},
                    dh::DofHandler, E::SymmetricTensor{4, dim},
                    u::Vector, ɛ_list, σ_list)

    global_dofs = zeros(Int, ndofs_per_cell(dh))
    ɛ = zero(SymmetricTensor{2, dim})
    σ = zero(SymmetricTensor{2, dim})
    Ω = 0.0
    for (cellcount, cell) in enumerate(CellIterator(dh))
        reinit!(cellvalues, cell)
        celldofs!(global_dofs, cell)
        uu = u[global_dofs]
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            ɛ_qp = symmetric(function_gradient(cellvalues, q_point, uu))
            σ_qp = E ⊡ ɛ_qp
            ɛ_list[cellcount] = ɛ_qp
            σ_list[cellcount] = σ_qp
            ɛ += ɛ_qp * dΩ
            σ += σ_qp * dΩ
            Ω += dΩ
        end
    end
    println(Ω)
    return ɛ, σ
end

end

import Homogenization
