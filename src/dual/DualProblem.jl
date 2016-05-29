@enum DualBlocksGlobal u◫ = 1 ξ⟂◫ = 2 ξo◫ = 3

@enum BlocksInner γ◫ = 1 τ◫ = 2
@enum BlocksOuter ε◫ = 1 χ⟂◫ = 2 χo◫ = 3



immutable DualGlobalProblem{dim, T, N}
    u_nodes::Vector{T}
    ξ⟂_nodes::Vector{Vector{T}}
    ξo_nodes::Vector{Vector{T}}

    f_u::PseudoBlockVector{T, Vector{T}}
    f_ξ⟂s::Vector{Vector{T}}
    f_ξos::Vector{Vector{T}}
    f::PseudoBlockVector{T, Vector{T}}

    K_uu::PseudoBlockMatrix{T, Matrix{T}}
    K_uξ⟂s::Vector{PseudoBlockMatrix{T, Matrix{T}}}
    K_uξos::Vector{PseudoBlockMatrix{T, Matrix{T}}}
    K_ξ⟂su::Vector{PseudoBlockMatrix{T, Matrix{T}}}
    K_ξ⟂sξ⟂s::Matrix{Matrix{T}}
    K_ξ⟂sξos::Matrix{Matrix{T}}
    K_ξosu::Vector{PseudoBlockMatrix{T, Matrix{T}}}
    K_ξosξ⟂s::Matrix{Matrix{T}}
    K_ξosξos::Matrix{Matrix{T}}

    K::PseudoBlockMatrix{T, Matrix{T}}

    # Dofs
    u_dofs::UnitRange{Int}
    ξ⟂s_dofs::Vector{UnitRange{Int}}
    ξos_dofs::Vector{UnitRange{Int}}

    # Misc
    χ⟂::Vector{T}
    χo::Vector{T}
    Aγεs::Vector{Tensor{2, dim, T, N}}

end

function reset!(dual_prob::DualGlobalProblem)
    @unpack dual_prob: f_u , f_ξ⟂s, f_ξos
    @unpack dual_prob: K_uu, K_uξ⟂s, K_uξos, K_ξ⟂su, K_ξ⟂sξ⟂s, K_ξ⟂sξos, K_ξosu, K_ξosξ⟂s, K_ξosξos
    # Reset the internal vectors to zero..
    fill_zero = x -> fill!(x, 0.0)
    fill!(f_u, 0.0)
    map(fill_zero, f_ξ⟂s)
    map(fill_zero, f_ξos)

    fill!(K_uu, 0.0)
    map(fill_zero, K_uξ⟂s)
    map(fill_zero, K_uξos)
    map(fill_zero, K_ξ⟂su)
    map(fill_zero, K_ξ⟂sξ⟂s)
    map(fill_zero, K_ξ⟂sξos)
    map(fill_zero, K_ξosu)
    map(fill_zero, K_ξosξ⟂s)
    map(fill_zero, K_ξosξos)
    return dual_prob
end

function DualGlobalProblem{dim}(nslips::Int, fspace_u::JuAFEM.FunctionSpace{dim},
                                fspace_ξ::JuAFEM.FunctionSpace{dim} = fspace_u)
    T = Float64

    nnodes = n_basefunctions(fspace_u)

    u = zeros(T, nnodes * dim)
    ξ⟂ = zeros(T, nnodes)
    ξo = zeros(T, nnodes)

    ξ⟂s = [similar(ξ⟂) for i in 1:nslips]
    ξos = [similar(ξ⟂) for i in 1:nslips]

    f_u  = PseudoBlockArray(zeros(T, dim * nnodes), dim * ones(Int, nnodes))
    f_ξ⟂s = [similar(ξ⟂) for i in 1:nslips]
    f_ξos = [similar(ξo) for i in 1:nslips]

    if dim == 2
        f = PseudoBlockArray(zeros(T, (nslips+dim) * nnodes), [length(u), [length(ξ⟂) for i in 1:nslips]...])
    else
        f = PseudoBlockArray(zeros(T, (2*nslips+dim) * nnodes), [length(u), [length(ξ⟂) for i in 1:nslips]..., [length(ξo) for i in 1:nslips]...])
    end

    if dim == 2
        @assert length(f_u) + sum(map(length, f_ξ⟂s))  == length(f)
    else
        @assert length(f_u) + sum(map(length, f_ξ⟂s)) + sum(map(length, f_ξos)) == length(f)
    end

    K_uu   = PseudoBlockArray(zeros(T, length(u), length(u)), dim * ones(Int, nnodes), dim * ones(Int, nnodes))
    K_uξ⟂s  = [PseudoBlockArray(zeros(T, length(u), length(ξ⟂)), dim * ones(Int, nnodes), ones(Int, nnodes)) for i in 1:nslips]
    K_uξos  = [PseudoBlockArray(zeros(T, length(u), length(ξo)), dim * ones(Int, nnodes), ones(Int, nnodes)) for i in 1:nslips]
    K_ξ⟂su  = [PseudoBlockArray(zeros(T, length(u), length(ξo)), dim * ones(Int, nnodes), ones(Int, nnodes)) for i in 1:nslips]
    K_ξ⟂sξ⟂s = [zeros(T, length(ξ⟂), length(ξ⟂)) for i in 1:nslips, j in 1:nslips]
    K_ξ⟂sξos = [zeros(T, length(ξ⟂), length(ξo)) for i in 1:nslips, j in 1:nslips]
    K_ξosu  = [PseudoBlockArray(zeros(T, length(ξo), length(u)), ones(Int, nnodes), dim * ones(Int, nnodes)) for i in 1:nslips]
    K_ξosξ⟂s = [zeros(T, length(ξo), length(ξ⟂)) for i in 1:nslips, j in 1:nslips]
    K_ξosξos = [zeros(T, length(ξ⟂), length(ξo)) for i in 1:nslips, j in 1:nslips]

    if dim == 2
        K = PseudoBlockArray(zeros(length(f), length(f)), [length(u), [length(ξ⟂) for i in 1:nslips]...],
                                                          [length(u), [length(ξ⟂) for i in 1:nslips]...])
    else
        K = PseudoBlockArray(zeros(length(f), length(f)), [length(u), nslips * length(ξ⟂), nslips * length(ξo)],
                                                          [length(u), nslips * length(ξ⟂), nslips * length(ξo)])
    end

    # Dofs
    ngradvars = dim - 1
    u_dofs = compute_udofs(dim, nnodes, ngradvars, nslips)

    if dim == 2
        ξ⟂_dofs = [compute_γdofs(dim, nnodes, ngradvars, nslips, α) for α in 1:nslips]
        ξo_dofs = [0:0 for α in 1:nslips]
    else
        ξ⟂_dofs = [compute_ξdofs(dim, nnodes, ngradvars, nslip, α, :ξ⟂) for α in 1:nslips]
        ξo_dofs = [compute_ξdofs(dim, nnodes, ngradvars, nslip, α, :ξo) for α in 1:nslips]
    end

    # Misc
    χ⟂ = zeros(T, nslips)
    χo = zeros(T, nslips)


    Aγεs = Vector{Tensor{2, dim, T, dim*dim}}(nslips)

    return DualGlobalProblem(u, ξ⟂s, ξos, # Nodal unknowns
                            f_u , f_ξ⟂s, f_ξos, f, # Forces
                            K_uu, K_uξ⟂s, K_uξos, K_ξ⟂su, K_ξ⟂sξ⟂s, K_ξ⟂sξos, K_ξosu, K_ξosξ⟂s, K_ξosξos, K, # Tangents
                            u_dofs, ξ⟂_dofs, ξo_dofs, # Dofs
                            χ⟂, χo, Aγεs) # Misc

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
    dR    = PseudoBlockArray(zeros(T, 2*nslips), [nslips, nslips])

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
