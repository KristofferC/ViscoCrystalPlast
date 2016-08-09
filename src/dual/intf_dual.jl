using ForwardDiff
using Parameters
using JuAFEM
using NLsolve

#include("local_problem.jl")

const problem = DualLocalProblem(2, Dim{2});

macro implement_jacobian(f, jacf)
    const N_INPUT = 6
    const N_OUTPUT = 4
    const CHUNK = 16
    const jacobian = zeros(N_INPUT, CHUNK)
    const result = Vector{ForwardDiff.Dual{CHUNK,Float64}}(N_OUTPUT)

    return quote
        function $(esc(f)){D<:ForwardDiff.Dual}(x::Vector{D})
            x_val = [ForwardDiff.value(z) for z in x]
            f_val, J = $(f)(x_val), $(jacf)(x_val)
            ForwardDiff.load_jacobian!($jacobian, x)
            J_new = J * $jacobian
            @assert length(f_val) == $(length(result))
            for i in eachindex(f_val)
                $result[i] = D(f_val[i], ForwardDiff.Partials(getrowpartials(D, J_new, i)))
            end
            return $result
        end
    end
end

@generated function getrowpartials{G<:ForwardDiff.Dual}(::Type{G}, J, i)
    return Expr(:tuple, [:(J[i, $k]) for k=1:ForwardDiff.npartials(G)]...)
end


################################################

function intf_dual{dim, T, Q, MS <:CrystPlastDualQD}(a::Vector{T}, prev_a::AbstractArray{Q},
                                                x::AbstractArray{Q}, fev::FEValues{dim}, fe_u, fe_ξ⟂, fe_ξo, dt,
                                                mss::AbstractVector{MS}, temp_mss::AbstractVector{MS}, mp::CrystPlastMP)
    @unpack mp: s, m, H⟂, Ee, sxm_sym, l
    nslip = length(sxm_sym)

    ngradvars = dim - 1
    n_basefuncs = n_basefunctions(get_functionspace(fev))
    nnodes = n_basefuncs

    @assert length(a) == nnodes * (dim + ngradvars * nslip)
    x_vec = reinterpret(Vec{dim, Q}, x, (n_basefuncs,))
    reinit!(fev, x_vec)


    fill!(fe_u, zero(Vec{dim, T}))
    for fe_ξ⟂_alpha in fe_ξ⟂
        fill!(fe_ξ⟂_alpha, zero(T))
    end
    if dim == 3
        for fe_ξo_alpha in fe_ξo
            fill!(fe_ξo_alpha, zero(T))
        end
    end


    χ⟂ = [zero(T) for i in 1:nslip]
    χo = [zero(T) for i in 1:nslip]

    q_rule = get_quadrule(fev)

    ud = compute_udofs(dim, nnodes, ngradvars, nslip)
    a_u = a[ud]
    u_nodes = reinterpret(Vec{dim, T}, a_u, (n_basefuncs,))
    ξ⟂_nodes = Vector{Vector{T}}(nslip)
    ξo_nodes = Vector{Vector{T}}(nslip)

    for α in 1:nslip
        if dim == 2
            ξ⟂_node_dofs = compute_γdofs(dim, nnodes, ngradvars, nslip, α)
        else
            ξ⟂_node_dofs = compute_ξdofs(dim, nnodes, 1, nslip, α, :ξ⟂)
            ξo_node_dofs = compute_ξdofs(dim, nnodes, 1, nslip, α, :ξo)
        end
        ξ⟂_nodes[α] = a[ξ⟂_node_dofs]
        if dim == 3
            ξo_nodes[α] = a[ξo_node_dofs]
        end
    end

    local ms
    local temp_ms
    _stress!(Y::Vector) = solve_local_problem(Y, problem, dt, mp, ms, temp_ms)
    _stress_diff(Y::Vector) = consistent_tangent(Y, problem, dt, mp, ms, temp_ms)
    @implement_jacobian _stress! _stress_diff

    for q_point in 1:length(JuAFEM.points(q_rule))
        ε = function_vector_symmetric_gradient(fev, q_point, u_nodes)

        for α in 1:nslip
            χ⟂[α] = function_scalar_gradient(fev, q_point, ξ⟂_nodes[α]) ⋅ mp.s[α]
            if dim == 3
                χo[α] = function_scalar_gradient(fev, q_point, ξo_nodes[α]) ⋅ mp.l[α]
            end
        end

        if dim == 2
            Y = [vec(ε); χ⟂]
        else
            Y = [vec(ε); χ⟂; χo]
        end
        ms = mss[q_point]
        temp_ms = temp_mss[q_point]
        X = _stress!(Y)

        γ = X[1:nslip]
        τ_di = X[nslip+1:end]

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
                temp_ms.χ⟂[α] = χ⟂[α]
                temp_ms.χo[α] = χo[α]
                temp_ms.τ_di[α] = τ_di[α]
                temp_ms.γ[α] = γ[α]
                temp_ms.τ[α] = (σ ⊡ mp.sxm_sym[α])
            end
        end

        for i in 1:n_basefuncs
            ∇ϕ = shape_gradient(fev, q_point, i)
            fe_u[i] += σ ⋅ ∇ϕ * detJdV(fev, q_point)
        end

        for α in 1:nslip
            ξ⟂_gp = function_scalar_value(fev, q_point, ξ⟂_nodes[α])
            g⟂_gp = ξ⟂_gp / (mp.H⟂ * mp.lα^2)

            if dim == 3
                ξo_gp = function_scalar_value(fev, q_point, ξo_nodes[α])
                go_gp = ξo_gp / (mp.Ho * mp.lα^2)
            end

            for i in 1:n_basefuncs
                ϕ = shape_value(fev, q_point, i)
                ∇ϕ = shape_gradient(fev, q_point, i)
                fe_ξ⟂[α][i] -= (g⟂_gp * ϕ + γ[α] * ∇ϕ ⋅ mp.s[α]) * detJdV(fev, q_point)
                if dim == 3
                    fe_ξo[α][i] -= (go_gp * ϕ + γ[α] * ∇ϕ ⋅ mp.l[α]) * detJdV(fev, q_point)
                end
            end
        end
    end

    fe = zeros(a)
    fe_u_jl = reinterpret(T, fe_u, (dim * n_basefuncs,))
    fe[ud] = fe_u_jl
    for α in 1:nslip
        if dim == 2
            fe[compute_γdofs(dim, nnodes, ngradvars, nslip, α)] = reinterpret(T, fe_ξ⟂[α], (n_basefuncs,))
        else
            fe[compute_ξdofs(dim, nnodes, 1, nslip, α, :ξ⟂)] = reinterpret(T, fe_ξ⟂[α], (n_basefuncs,))
            fe[compute_ξdofs(dim, nnodes, 1, nslip, α, :ξo)] = reinterpret(T, fe_ξo[α], (n_basefuncs,))
        end
    end
    return fe
end
