using Parameters
using JuAFEM
using ContMechTensors

if !isdefined(:CrystPlastMS)
    @eval begin
        type CrystPlastMS{dim, T, M}
            σ::SymmetricTensor{2, dim, T, M}
            ε::SymmetricTensor{2, dim, T, M}
            ε_p::SymmetricTensor{2, dim, T, M}
            τ_di::Vector{T}
            τ::Vector{T}
            g::Vector{Vec{dim, T}}
        end
    end
end


function CrystPlastMS(nslip, dim)
    σ = zero(SymmetricTensor{2, dim})
    ε = zero(SymmetricTensor{2, dim})
    ε_p = zero(SymmetricTensor{2, dim})
    τ_di = zeros(nslip)
    τ = zeros(nslip)
    g = Vec{dim, Float64}[zero(Vec{dim, Float64}) for i in 1:nslip]
    return CrystPlastMS(σ, ε, ε_p, τ_di, τ, g)
end


function intf_opt{dim, T, Q}(a::Vector{T}, a_prev, x::AbstractArray{Q}, fev::FEValues{dim}, dt, mss::AbstractVector{CrystPlastMS}, mp::CrystPlastMP)
    @unpack mp: s, m, l, H⟂, Ho, Ee, sxm_sym
    nslip = length(sxm_sym)


    ngradvars = 1
    n_basefuncs = n_basefunctions(get_functionspace(fev))
    nnodes = n_basefuncs

    @assert length(a) == nnodes * (dim + ngradvars * nslip)
    @assert length(a_prev) == nnodes * (dim + ngradvars * nslip)

    x_vec = reinterpret(Vec{dim, Q}, x, (n_basefuncs,))
    reinit!(fev, x_vec)

    fe_u = [zero(Vec{dim, T}) for i in 1:n_basefuncs]
    fe_g = [zeros(T, n_basefuncs)  for i in 1:ngradvars*nslip]


    q_rule = get_quadrule(fev)

    ud = u_dofs(dim, nnodes, ngradvars, nslip)
    a_u = a[ud]
    u_vec = reinterpret(Vec{dim, T}, a_u, (n_basefuncs,))
    γs = Vector{Vector{T}}(nslip)
    γs_prev = Vector{Vector{T}}(nslip)
    for α in 1:nslip
        gd = g_dofs(dim, nnodes, ngradvars, nslip, α)
        γs[α] = a[gd]
        γs_prev[α] = a_prev[gd]
    end

    H_g = [H⟂ * s[α] ⊗ s[α] + Ho * l[α] ⊗ l[α] for α in 1:nslip]

    for q_point in 1:length(JuAFEM.points(q_rule))
        ε = function_vector_symmetric_gradient(fev, q_point, u_vec)
        ε_p = zero(SymmetricTensor{2, dim, T})

        for α in 1:nslip
            γ = function_scalar_value(fev, q_point, γs[α])
            # displacements
            ε_p += γ * sxm_sym[α]
        end

        ε_e = ε - ε_p
        σ = Ee * ε_e
        for i in 1:n_basefuncs
            fe_u[i] +=  σ ⋅ shape_gradient(fev, q_point, i) * detJdV(fev, q_point)
        end

        if T == Float64
            mss[q_point].σ  = σ
            mss[q_point].ε  = ε
            mss[q_point].ε_p = ε_p
        end

        for α in 1:nslip
            γ = function_scalar_value(fev, q_point, γs[α])
            γ_prev = function_scalar_value(fev, q_point, γs_prev[α])

            τα = compute_tau(γ, γ_prev, dt, mp)
            τ_en = -(σ ⊡ sxm_sym[α])

            g = function_scalar_gradient(fev, q_point, γs[α])
            ξ = mp.lα^2 * H_g[α] * g
            for i in 1:n_basefuncs
                fe_g[α][i] += (shape_value(fev, q_point, i) * (τα + τ_en) +
                               shape_gradient(fev, q_point, i) ⋅ ξ) * detJdV(fev, q_point)
            end

            if T == Float64
                mss[q_point].g[α] = g
                mss[q_point].τ_di[α] = τα
                mss[q_point].τ[α] = -τ_en
            end
        end
    end

    fe = zeros(a)
    fe_u_jl = reinterpret(T, fe_u, (dim * n_basefuncs,))

    fe[ud] = fe_u_jl
    for α in 1:nslip
        fe[g_dofs(dim, nnodes, ngradvars, nslip, α)] = fe_g[α]
    end

    return fe
end

function compute_tau(γ_gp, γ_gp_prev, ∆t, mp::CrystPlastMP)
    @unpack mp: C, tstar, n
    Δγ = γ_gp - γ_gp_prev
    τ = C * (tstar / ∆t * abs(Δγ))^(1/n)
    return sign(Δγ) * τ
end
