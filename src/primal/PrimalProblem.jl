@enum PrimalBlocksGlobal up◫ = 1 γp◫ = 2

immutable PrimalGlobalProblem{T}
    u_nodes::Vector{T}
    γ_nodes::Vector{Vector{T}}
    γ_prev_nodes::Vector{Vector{T}}

    f_u::PseudoBlockVector{T, Vector{T}}
    f_γs::Vector{Vector{T}}
    f::PseudoBlockVector{T, Vector{T}}

    K_uu::PseudoBlockMatrix{T, Matrix{T}}
    K_uγs::Vector{PseudoBlockMatrix{T, Matrix{T}}}
    K_γsu::Vector{PseudoBlockMatrix{T, Matrix{T}}}
    K_γsγs::Matrix{Matrix{T}}

    K::PseudoBlockMatrix{T, Matrix{T}}

    # Dofs
    u_dofs::UnitRange{Int}
    γ_dofs::Vector{UnitRange{Int}}

    # Misc
    Aτγ::Vector{T}

end

function reset!(primal_prob::PrimalGlobalProblem)
    @unpack primal_prob: f_u , f_γs
    @unpack primal_prob: K_uu, K_uγs, K_γsu, K_γsγs
    # Reset the internal vectors to zero..
    fill_zero = x -> fill!(x, 0.0)
    fill!(f_u, 0.0)
    map(fill_zero, f_γs)

    fill!(K_uu, 0.0)
    map(fill_zero, K_uγs)
    map(fill_zero, K_γsu)
    map(fill_zero, K_γsγs)

    return primal_prob
end

function PrimalGlobalProblem{dim}(nslips::Int, fspace_u::JuAFEM.FunctionSpace{dim},
                                 fspace_ξ::JuAFEM.FunctionSpace{dim} = fspace_u)
    T = Float64

    nnodes = n_basefunctions(fspace_u)

    u = zeros(T, nnodes * dim)
    γ = zeros(T, nnodes)
    γ_prev = zeros(T, nnodes)

    γs = [similar(γ) for i in 1:nslips]
    γ_prevs = [similar(γ_prev) for i in 1:nslips]

    f_u  = PseudoBlockArray(zeros(T, dim * nnodes), dim * ones(Int, nnodes))
    f_γs = [similar(γ) for i in 1:nslips]
    f = PseudoBlockArray(zeros(T, (nslips+dim) * nnodes), [length(u), [length(γ) for i in 1:nslips]...])

    @assert length(f_u) + sum(map(length, f_γs))  == length(f)

    K_uu   = PseudoBlockArray(zeros(T, length(u), length(u)), dim * ones(Int, nnodes), dim * ones(Int, nnodes))
    K_uγs  = [PseudoBlockArray(zeros(T, length(u), length(γ)), dim * ones(Int, nnodes), ones(Int, nnodes)) for i in 1:nslips]
    K_γsu  = [PseudoBlockArray(zeros(T, length(u), length(γ)), dim * ones(Int, nnodes), ones(Int, nnodes)) for i in 1:nslips]
    K_γsγs = [zeros(T, length(γ), length(γ)) for i in 1:nslips, j in 1:nslips]


    K = PseudoBlockArray(zeros(length(f), length(f)), [length(u), [length(γ) for i in 1:nslips]...],
                                                      [length(u), [length(γ) for i in 1:nslips]...])


    # Dofs
    ngradvars = 1
    u_dofs = compute_udofs(dim, nnodes, ngradvars, nslips)
    γ_dofs = [compute_γdofs(dim, nnodes, ngradvars, nslips, α) for α in 1:nslips]

    # Misc
    Aτγ = zeros(T, nslips)

    return PrimalGlobalProblem(u, γs, γ_prevs,  # Nodal unknowns
                               f_u , f_γs, f, # Forces
                               K_uu, K_uγs, K_γsu, K_γsγs, K, # Tangents
                               u_dofs, γ_dofs, # Dofs
                               Aτγ) # Misc

end
