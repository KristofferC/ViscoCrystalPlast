@enum DualBlocksGlobal u = 1 ξ⟂ = 2 ξo = 3

if !(isdefined(:DualGlobalProblem))
    @eval begin
    immutable DualGlobalProblem{dim, T}
    u::Vector{T}
    ξ⟂::Vector{T}
    ξo::Vector{T}
    a::PseudoBlockVector{T, Vector{T}}

    fu::Vector{T}
    fξ⟂::Vector{T}
    fξ⟂o::Vector{T}
    f::PseudoBlockVector{T, Vector{T}}

    Kuu::Matrix{T}
    Kuξ⟂::Matrix{T}
    Kuξo::Matrix{T}
    Kξ⟂u::Matrix{T}
    Kξ⟂ξ⟂::Matrix{T}
    Kξ⟂ξo::Matrix{T}
    Kξou::Matrix{T}
    Kξoξ⟂::Matrix{T}
    Kξoξo::Matrix{T}

    K::PseudoBlockMatrix{T, Matrix{T}}
    end
    end
end

function DualGlobalProblem{dim}(nslips::Int, fspace_u::JuAFEM.FunctionSpace{dim},
                                fspace_ξ::JuAFEM.FunctionSpace{dim} = fspace_u)
    T = Float64

    u_ = zeros(T, n_basefunctions(fspace_u) * dim)
    ξ_⟂ = zeros(T, n_basefunctions(fspace_u))
    ξ_o = zeros(T, n_basefunctions(fspace_u))
    a = PseudoBlockArray(zeros(T, (2+dim) * n_basefunctions(fspace_u)), [length(u), length(ξ⟂), length(ξo)])

    f_u  = similar(u)
    f_ξ⟂ = similar(ξ⟂)
    f_ξo = similar(ξo)
    f = similar(a)

    K_uu   = zeros(T, length(u), length(u))
    K_uξ⟂  = zeros(T, length(u) , length(ξ⟂))
    K_uξo  = zeros(T, length(u) , length(ξo))
    K_ξ⟂u  = zeros(T, length(ξ⟂), length(u))
    K_ξ⟂ξ⟂ = zeros(T, length(ξ⟂), length(ξ⟂))
    K_ξ⟂ξo = zeros(T, length(ξ⟂), length(ξo))
    K_ξou  = zeros(T, length(ξo), length(u))
    K_ξoξ⟂ = zeros(T, length(ξo), length(ξ⟂))
    K_ξoξo = zeros(T, length(ξo), length(ξo))

    K = PseudoBlockArray(zeros(length(a), length(a)), [length(u), length(ξ⟂), length(ξo)], [length(u), length(ξ⟂), length(ξo)])

    return DualGlobalProblem{dim, T}(u, ξ_⟂, ξ_o, a, f_u , f_ξ⟂, f_ξo, f, K_uu, K_uξ⟂, K_uξo,
                                     K_ξ⟂u, K_ξ⟂ξ⟂, K_ξ⟂ξo, K_ξou, K_ξoξ⟂, K_ξoξo, K)
end



function intf{dim, T, Q, MS <:CrystPlastDualQD}(a::Vector{T}, prev_a::AbstractArray{Q}, x::AbstractArray{Q}, fev::FEValues{dim}, fe_u, fe_g, dt,
                             mss::AbstractVector{MS}, temp_mss::AbstractVector{MS}, mp::CrystPlastMP)
    @unpack mp: s, m, H⟂, Ee, sxm_sym, l
    nslip = length(sxm_sym)

    ndim = 2
    ngradvars = 1
    n_basefuncs = n_basefunctions(get_functionspace(fev))
    nnodes = n_basefuncs

    @assert length(a) == nnodes * (ndim + ngradvars * nslip)
    x_vec = reinterpret(Vec{2, Q}, x, (n_basefuncs,))
    reinit!(fev, x_vec)


    fill!(fe_u, zero(Vec{dim, T}))
    for fe_g_alpha in fe_g
        fill!(fe_g_alpha, zero(T))
    end

    χ = [zero(T) for i in 1:nslip]
    ξα = [zero(Vec{2, T}) for i in 1:n_basefuncs]

    ϕ = zeros(n_basefuncs)
    dV = zeros(n_basefuncs)
    ∇ϕ = [zeros(Vec{2, T}) for i in 1:n_basefuncs]

    q_rule = get_quadrule(fev)

    ud = u_dofs(ndim, nnodes, ngradvars, nslip)
    a_u = a[ud]
    u_nodes = reinterpret(Vec{2, T}, a_u, (n_basefuncs,))
    ξ⟂_nodes = Vector{Vector{T}}(nslip)
    for α in 1:nslip
        ξ⟂_node_dofs = g_dofs(ndim, nnodes, ngradvars, nslip, α)
        ξ⟂_nodes[α] = reinterpret(T, a[ξ⟂_node_dofs], (n_basefuncs,))
    end

    local ms
    local temp_ms
    _stress!(Y::Vector) = stress!(Y, dt, mp, ms, temp_ms)
    _stress_diff(Y::Vector) = consistent_tangent(Y, dt, mp, temp_ms)
    @implement_jacobian _stress! _stress_diff

    for q_point in 1:length(JuAFEM.points(q_rule))
        ε = function_vector_symmetric_gradient(fev, q_point, u_nodes)

        for α in 1:nslip
            χ⟂[α] = function_scalar_gradient(fev, q_point, ξ⟂_nodes[α]) ⋅ mp.s[α]
            if dim == 3
                χo[α] = function_scalar_gradient(fev, q_point, ξo_nodes[α]) ⋅ mp.l[α]
            end
        end

        out = [vec(ε); χ]
        ms = mss[q_point]
        temp_ms = temp_mss[q_point]
        inner = solve_local_problem(Y, local_problem, ∆t, mp, ms, temp_ms)
        γ = getblock!(inner.γ, inner, γ◫)
        τ_di = getblock!(inner.τ, inner, τ◫)
        A = consistent_tangent(out, local_problem, ∆t, mp, ms, temp_ms)

        # Store
        ε_p = zero(ε)
        for α in 1:nslip
            ε_p += γ[α] * mp.sxm_sym[α]
        end

        ε_e = ε - ε_p
        σ = mp.Ee ⊡ ε_e

        if T == Float64
            temp_ms.σ  = σ
            temp_ms.ε  = ε
            temp_ms.ε_p = ε_p
            for α in 1:nslip
                temp_ms.χ[α] = χ[α]
                temp_ms.τ_di[α] = τ_di[α]
                temp_ms.γ[α] = γ[α]
                temp_ms.τ[α] = (σ ⊡ mp.sxm_sym[α])
            end
        end

        for i in 1:n_basefuncs
            ϕ[i] = shape_value(fev, q_point, i)
            ∇ϕ[i] = shape_gradient(fev, q_point, i)
            dV[i] = detJdV(fev, q_point, i)
        end

        for α in 1:nslips
            for β in 1:nslips
                for i in 1:n_basefuncs
                    for j in 1:n_basefuncs
                        if α == β
                            K_ξoξo[i,j] += ϕ[i] / (mp.H⟂ * mp.lα^2) * ϕ[j] * dV[i]
                        end

        for i in 1:n_basefuncs
            ∇ϕ = shape_gradient(fev, q_point, i)
            fe_u[i] += σ ⋅ ∇ϕ * detJdV(fev, q_point)

            for j in 1:n_basefuncs

        end

        for α in 1:nslip
            ξ⟂_gp = function_scalar_value(fev, q_point, ξ⟂_nodes[α])
            g⟂_gp = ξ⟂_gp / (mp.H⟂ * mp.lα^2)

            for i in 1:n_basefuncs

                ϕ = shape_value(fev, q_point, i)
                ∇ϕ = shape_gradient(fev, q_point, i)
                fe_g[α][i] += (g⟂_gp * ϕ + γ[α] * ∇ϕ ⋅ mp.s[α]) * detJdV(fev, q_point)
            end
        end
    end

    fe = zeros(a)
    fe_u_jl = reinterpret(T, fe_u, (ndim * n_basefuncs,))
    fe[ud] = fe_u_jl
    for α in 1:nslip
        fe[g_dofs(ndim, nnodes, ngradvars, nslip, α)] = reinterpret(T, fe_g[α], (n_basefuncs,))
    end
    return fe
end
