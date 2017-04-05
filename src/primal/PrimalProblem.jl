@enum PrimalBlocksGlobal up◫ = 1 γp◫ = 2

immutable PrimalGlobalProblem{dim, T, N}
    u_nodes::Vector{T}
    γ_nodes::Vector{Vector{T}}
    γ_prev_nodes::Vector{Vector{T}}

    f::Vector{T}
    K::Matrix{T}

    # Dofs
    u_dofs::UnitRange{Int}
    γ_dofs::Vector{UnitRange{Int}}

    # Misc
    Aτγ::Vector{T}
    δε::Vector{SymmetricTensor{2, dim, T, N}}
end


function PrimalGlobalProblem{dim}(nslips::Int, fev_u::CellVectorValues{dim}, fev_γ::CellScalarValues{dim})
    T = Float64

    nbasefuncs_γ = getnbasefunctions(fev_γ)
    nbasefuncs_u = getnbasefunctions(fev_u)
    nnodes = nbasefuncs_γ

    u      = zeros(T, nbasefuncs_u)
    γ      = zeros(T, nbasefuncs_γ)
    γ_prev = zeros(T, nbasefuncs_γ)

    γs      = [similar(γ)      for i in 1:nslips]
    γ_prevs = [similar(γ_prev) for i in 1:nslips]

    n_total_dofs = nbasefuncs_u + nbasefuncs_γ * nslips
    f = zeros(T, n_total_dofs)
    K = zeros(T, n_total_dofs, n_total_dofs)

    # Dofs
    ngradvars = 1
    u_dofs = compute_udofs(dim, nnodes, ngradvars, nslips)
    γ_dofs = [compute_γdofs(dim, nnodes, ngradvars, nslips, α) for α in 1:nslips]

    # Misc
    Aτγ = zeros(T, nslips)
    δε = [zero(SymmetricTensor{2, dim, T}) for i in 1:nbasefuncs_u]

    return PrimalGlobalProblem(u, γs, γ_prevs,  # Nodal unknowns
                               f, # Forces
                               K, # Tangents
                               u_dofs, γ_dofs, # Dofs
                               Aτγ, δε) # Misc

end
