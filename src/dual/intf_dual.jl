using ForwardDiff
using Parameters
using JuAFEM
using NLsolve


macro implement_jacobian(f, jacf)
    const GG = ForwardDiff.workvec_eltype(ForwardDiff.GradientNumber, Float64, Val{12}, Val{12})
    const result = ForwardDiff.build_workvec(GG, 6)
    return quote
        function $(esc(f)){G<:ForwardDiff.GradientNumber}(x::Vector{G})
            x_val = ForwardDiff.get_value(x)
            f_val, J = $(f)(x_val), $(jacf)(x_val)
            J_new = J * ForwardDiff.get_jacobian(x)

            for i in eachindex(f_val)
                $result[i] = G(f_val[i], getrowpartials(G, J_new, i))
            end
            return $result
        end
    end
end

@generated function getrowpartials{G<:ForwardDiff.GradientNumber}(::Type{G}, J, i)
    return Expr(:tuple, [:(J[i, $k]) for k=1:ForwardDiff.npartials(G)]...)
end

################################################

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
            χ[α] = function_scalar_gradient(fev, q_point, ξ⟂_nodes[α]) ⋅ mp.s[α]
        end

        Y = [vec(ε); χ]
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
        σ = mp.Ee * ε_e


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
            ϕ = shape_value(fev, q_point, i)
            ∇ϕ = shape_gradient(fev, q_point, i)
            fe_u[i] += σ ⋅ ∇ϕ * detJdV(fev, q_point)
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


const dummycache = ForwardDiffCache()

function stress!{T}(Y::Vector{T}, ∆t, mp, ms, temp_ms)
    R!(r, x) = compute_residual!(r, x, Y, ∆t, mp, ms)
    nslip = length(mp.angles)
    x0 = zeros(2*nslip)

    for α in 1:nslip
        x0[α] = ms.γ[α]
        x0[α+nslip] = ms.τ_di[α]
    end

    J = jacobian(R!, output_length = length(x0), cache=dummycache)

    r = ones(2*nslip)
    jbuf = zeros(2*nslip, 2*nslip)

    max_iters = 40
    n_iters = 1


    while norm(r, Inf) >= 1e-7
         compute_residual!(r, x0, Y, ∆t, mp, ms)
         compute_jacobian!(jbuf, x0, Y, ∆t, mp, ms)

         x0 -= jbuf\ r
         if n_iters == max_iters
            error("Non conv mat")
        end
        n_iters +=1
    end

    γ = x0[1:nslip]
    τ_di = x0[nslip+1:end]
    return x0
end

function compute_stress(ε, γ, sxm_sym, Ee)
    ε_p = zero(ε)
    for α in 1:length(γ)
        ε_p += γ[α] * sxm_sym[α]
    end

    ε_e = ε - ε_p
    σ = Ee * ε_e
    return σ
end



function compute_residual!(r, X, Y, Δt, mp, ms)
    nslip = length(mp.angles)
    ε = convert(SymmetricTensor, Tensor{2, 2}(Y[1:4]))
    χs = Y[5:end]
    γ = X[1:nslip]
    τ_di = X[nslip+1:end]

    σ = compute_stress(ε, γ, mp.sxm_sym, mp.Ee)

    for α in 1:nslip
        τen = -(σ ⊡ mp.sxm_sym[α])
        Δγ = γ[α] - ms.γ[α]
        r[α] = τen + τ_di[α] - χs[α]
        r[α+nslip] = Δγ - Δt / mp.tstar * (abs(τ_di[α]) / mp.C )^(mp.n) * sign(τ_di[α])
    end
    return
end

function compute_jacobian!(J, X, Y, Δt, mp, ms)
    fill!(J, 0.0)
    nslip = length(mp.angles)

    @inbounds for α in 1:nslip
        for β in 1:nslip
            J[α, β] = mp.Dαβ[α, β]
        end

        τ_di = X[nslip+α]

        J[α, nslip + α] = 1.0
        J[nslip + α, α] = 1.0
        J[nslip + α, nslip + α] = - Δt / mp.tstar * mp.n / mp.C * (abs(τ_di) / mp.C )^(mp.n-1)
    end
end


function consistent_tangent(Y, ∆t, matpar, temp_matstat)
    nslip = length(matpar.angles)
    X = zeros(2*nslip)
    drdX = zeros(2*nslip, 2*nslip)
    drdY = zeros(2*nslip, 4 + nslip)
    for α in 1:nslip
        X[α] = temp_matstat.γ[α]
        X[α+nslip] = temp_matstat.τ_di[α]
    end

    for α in 1:nslip
        drdY[α, 1:length(matpar.Esm[α])] = -matpar.Esm[α]
        drdY[α, α + length(matpar.Esm[α])] = -1.0
    end

    compute_jacobian!(drdX, X, Y, ∆t, matpar, temp_matstat)

    dXdY = - inv(drdX) * drdY

    return dXdY
end
