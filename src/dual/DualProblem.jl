const u◫ = Block(1); const ξ⟂◫ = Block(2); const ξo◫ = Block(3)
const γ◫ = Block(1); const τ◫ = Block(2)
const ε◫ = Block(1); const χ⟂◫ = Block(2); const χo◫ = Block(3)

immutable DualGlobalProblem{dim, T, N}
    u_nodes::Vector{T}
    ξ⟂_nodes::Vector{Vector{T}}
    ξo_nodes::Vector{Vector{T}}

    f::Vector{T}
    C_f::Vector{T}
    K::Matrix{T}
    C_K::Matrix{T}

    # Dofs
    u_dofs::UnitRange{Int}
    ξ⟂s_dofs::Vector{UnitRange{Int}}
    ξos_dofs::Vector{UnitRange{Int}}

    # Misc
    χ⟂::Vector{T}
    χo::Vector{T}
    Aγεs::Vector{SymmetricTensor{2, dim, T, N}}
    δε::Vector{SymmetricTensor{2, dim, T, N}}
    DAγξ⟂s::Vector{SymmetricTensor{2, dim, T, N}}
    DAγξos::Vector{SymmetricTensor{2, dim, T, N}}
end


function DualGlobalProblem{dim}(nslips::Int, fev_u::CellVectorValues{dim}, fev_ξ::CellScalarValues{dim})
    T = Float64

    nbasefuncs_ξ = getnbasefunctions(fev_ξ)
    nbasefuncs_u = getnbasefunctions(fev_u)
    nnodes = nbasefuncs_ξ

    u = zeros(T, nbasefuncs_u)
    ξ⟂= zeros(T, nbasefuncs_ξ)
    ξo= zeros(T, nbasefuncs_ξ)

    ξ⟂s = [similar(ξ⟂) for i in 1:nslips]
    ξos = [similar(ξ⟂) for i in 1:nslips]

    n_ξ_slip = nbasefuncs_ξ * nslips
    n_tot_ξ = n_ξ_slip * (dim - 1)
    n_total_dofs = nbasefuncs_u + n_tot_ξ

    f = zeros(T, n_total_dofs)
    C_f = zeros(T, dim * dim)
    K = zeros(T, n_total_dofs, n_total_dofs)
    C_K = zeros(T, nbasefuncs_u, dim * dim)

    # Dofs
    ngradvars = dim - 1
    u_dofs = compute_udofs(dim, nnodes, ngradvars, nslips)

    if dim == 2
        ξ⟂_dofs = [compute_γdofs(dim, nnodes, 1, nslips, α) for α in 1:nslips]
        ξo_dofs = [0:0 for α in 1:nslips]
    else
        ξ⟂_dofs = [compute_ξdofs(dim, nnodes, 1, nslips, α, :ξ⟂) for α in 1:nslips]
        ξo_dofs = [compute_ξdofs(dim, nnodes, 1, nslips, α, :ξo) for α in 1:nslips]
    end

    # Misc
    χ⟂ = zeros(T, nslips)
    χo = zeros(T, nslips)


    Aγεs = [zero(SymmetricTensor{2, dim, T}) for i in 1:nslips]

    δε     = [zero(SymmetricTensor{2, dim, T}) for i in 1:nbasefuncs_u]
    DAγξ⟂s = [zero(SymmetricTensor{2, dim, T}) for α in 1:nslips]
    DAγξos = [zero(SymmetricTensor{2, dim, T}) for α in 1:nslips]

    return DualGlobalProblem(u, ξ⟂s, ξos, # Nodal unknowns
                            f, C_f # Forces
                            K, C_K # Tangents
                            u_dofs, ξ⟂_dofs, ξo_dofs, # Dofs
                            χ⟂, χo, Aγεs, δε, DAγξ⟂s, DAγξos) # Misc

end


immutable DualLocalProblem{dim, T}
    # Inner
    γ::Vector{T}
    τ::Vector{T}

    # Outer
    ε::Vector{T}
    χ⟂::Vector{T}
    χo::Vector{T}

    # Residual
    J_ττ::Matrix{T}
    J_τγ::Matrix{T}
    J_γτ::Matrix{T}
    J_γγ::Matrix{T}
    J::PseudoBlockMatrix{T, Matrix{T}}
    R_τ::Vector{T}
    R_γ::Vector{T}
    R::PseudoBlockVector{T, Vector{T}}


    # ATS tensor
    Q_γε ::Matrix{T}
    Q_γχ⟂::Matrix{T}
    Q_γχo::Matrix{T}
    Q_τε ::Matrix{T}
    Q_τχ⟂::Matrix{T}
    Q_τχo::Matrix{T}
    Q::PseudoBlockMatrix{T, Matrix{T}}

    A::BlockMatrix{T, Matrix{T}}

    outer::PseudoBlockVector{T, Vector{T}}
    inner::PseudoBlockVector{T, Vector{T}}
end



function DualLocalProblem{dim}(nslips::Int, ndim::Type{Dim{dim}})
    T = Float64

    if dim == 2
        ncomp = 4
    elseif dim == 3
        ncomp = 9
    else
        error("invalid dim")
    end

    γ = zeros(T, nslips)
    τ = zeros(T, nslips)
    ε = zeros(T, ncomp)
    χ⟂ = zeros(T, nslips)
    χo = zeros(T, nslips)

    J_ττ = zeros(T, nslips, nslips)
    J_τγ = zeros(T, nslips, nslips)
    J_γτ = zeros(T, nslips, nslips)
    J_γγ = zeros(T, nslips, nslips)
    J    = PseudoBlockArray(zeros(T, 2*nslips, 2*nslips), [nslips, nslips], [nslips, nslips])
    R_τ  = zeros(T, nslips)
    R_γ  = zeros(T, nslips)
    R    = PseudoBlockArray(zeros(T, 2*nslips), [nslips, nslips])
    dR   = PseudoBlockArray(zeros(T, 2*nslips), [nslips, nslips])

    Q_γε = zeros(T, nslips, ncomp)
    Q_γχ⟂ = zeros(T, nslips, nslips)
    Q_γχo = zeros(T, nslips, nslips)
    Q_τε = zeros(T, nslips, ncomp)
    Q_τχ⟂ = zeros(T, nslips, nslips)
    Q_τχo = zeros(T, nslips, nslips)

    if dim == 2
        Q = PseudoBlockArray(zeros(T, 2*nslips, ncomp+nslips), [nslips, nslips], [ncomp, nslips])
        outer = PseudoBlockArray(zeros(T, ncomp+nslips), [ncomp, nslips])
        A = BlockArray(zeros(T, 2*nslips, ncomp+nslips), [nslips, nslips], [ncomp, nslips])
    else
        Q = PseudoBlockArray(zeros(T, 2*nslips, ncomp+2*nslips), [nslips, nslips], [ncomp, nslips, nslips])
        outer = PseudoBlockArray(zeros(T, ncomp+2*nslips), [ncomp, nslips, nslips])
        A = BlockArray(zeros(T, 2*nslips, ncomp+2*nslips), [nslips, nslips], [ncomp, nslips, nslips])
    end
    inner = PseudoBlockArray(zeros(T, 2*nslips), [nslips, nslips])
    Δinner = PseudoBlockArray(zeros(T, 2*nslips), [nslips, nslips])
    DualLocalProblem{dim, T}(γ, τ, ε, χ⟂, χo, J_ττ, J_τγ, J_γτ, J_γγ, J, R_τ, R_γ, R,
                             Q_γε, Q_γχ⟂, Q_γχo, Q_τε, Q_τχ⟂, Q_τχo, Q, A, outer, inner)
end

function reset!(dlp::DualLocalProblem)
    fill!(full(dlp.J), 0.0)
    fill!(full(dlp.R), 0.0)
    fill!(full(dlp.Q), 0.0)
    return dlp
end
