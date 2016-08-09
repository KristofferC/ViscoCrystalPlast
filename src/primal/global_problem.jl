

function intf{dim, func_space, T, Q, QD <: CrystPlastPrimalQD}(primal_prob::PrimalProblem,
                           a::Vector{T}, a_prev, x::Vector, fev::FEValues{dim, Q, func_space},
                            dt, mss::AbstractVector{QD}, temp_mss::AbstractVector{QD}, mp::CrystPlastMP)


    @unpack mp: s, m, l, H⟂, Ho, Ee, sxm_sym

    glob_prob = primal_prob.global_problem

    @unpack glob_prob: u_nodes, γ_nodes, γ_prev_nodes
    @unpack glob_prob: f_u , f_γs, f
    @unpack glob_prob: K_uu, K_uγs, K_γsu, K_γsγs, K
    @unpack glob_prob: u_dofs, γ_dofs
    @unpack glob_prob: Aτγ


    nslip = length(sxm_sym)

    ngradvars = 1
    nnodes = n_basefunctions(get_functionspace(fev))

    @assert length(a) == nnodes * (dim + ngradvars * nslip)
    @assert length(a_prev) == nnodes * (dim + ngradvars * nslip)

    #x_vec = reinterpret(Vec{dim, Q}, x, (nnodes,))
    reinit!(fev, x)
    reset!(glob_prob)

    extract!(u_nodes, a, u_dofs)

    u_vec = reinterpret(Vec{dim, T}, u_nodes, (nnodes,))

    for α in 1:nslip
        extract!(γ_nodes[α], a, γ_dofs[α])
        extract!(γ_prev_nodes[α], a_prev, γ_dofs[α])
    end

    @inbounds begin

    for q_point in 1:length(points(get_quadrule(fev)))
        ϕ = i -> shape_value(fev, q_point, i)
        ∇ϕ = i -> shape_gradient(fev, q_point, i)
        ε = function_vector_symmetric_gradient(fev, q_point, u_vec)
        ε_p = zero(SymmetricTensor{2, dim, T})

        for α in 1:nslip
            γ = function_scalar_value(fev, q_point, γ_nodes[α])
            ε_p += γ * sxm_sym[α]
        end

        ε_e = ε - ε_p
        σ = Ee ⊡ ε_e
        for i in 1:nnodes
            updateblock!(f_u, ∇ϕ(i) ⋅ σ * detJdV(fev, q_point), +, i)
            for j in 1:nnodes
                K_uu_gp = dotdot(∇ϕ(i), Ee, ∇ϕ(j)) * detJdV(fev, q_point)
                updateblock!(K_uu, K_uu_gp, +, i, j)
                for β in 1:nslip
                    K_uγ_qp = - ∇ϕ(i) ⋅ mp.Esm[β] * ϕ(j) * detJdV(fev, q_point)
                    #K_γu_qp = - ϕ(j) * mp.Esm[β] ⋅ ∇ϕ(i) * detJdV(fev, q_point)
                    updateblock!(K_uγs[β], K_uγ_qp, +, i, j)
                    updateblock!(K_γsu[β], K_uγ_qp, +, i, j) # sym
                    #updateblock!(K_γsu[β], K_γu_qp, +, i, j)
                end
            end
        end

        if T == Float64
            temp_mss[q_point].σ  = σ
            temp_mss[q_point].ε  = ε
            temp_mss[q_point].ε_p = ε_p
        end

        for α in 1:nslip
            γ = function_scalar_value(fev, q_point, γ_nodes[α])
            γ_prev = function_scalar_value(fev, q_point, γ_prev_nodes[α])
            τα = compute_tau(γ, γ_prev, dt, mp)
            τ_en = -(σ ⊡ sxm_sym[α])
            Aτγ[α] = diff_tau(γ, γ_prev, dt, mp)

            g = function_scalar_gradient(fev, q_point, γ_nodes[α])
            ξ = mp.lα^2 * mp.Hgrad[α] ⋅ g
            for i in 1:nnodes
                f_γs[α][i] += (ϕ(i) * (τα + τ_en) + ∇ϕ(i) ⋅ ξ) * detJdV(fev, q_point)
            end

            if T == Float64
                temp_mss[q_point].ξ⟂[α] = g ⋅ mp.s[α] * mp.lα^2 * mp.H⟂
                temp_mss[q_point].ξo[α] = g ⋅ mp.l[α] * mp.lα^2 * mp.Ho
                temp_mss[q_point].τ_di[α] = τα
                temp_mss[q_point].τ[α] = -τ_en
            end
        end

        for α in 1:nslip, β in 1:nslip
            K_γsγsαβ = K_γsγs[α, β]
            for i in 1:nnodes, j in 1:nnodes
                K_γsγsαβ[i,j] += ϕ(i) * mp.Dαβ[α, β] * ϕ(j) * detJdV(fev, q_point)
                if α == β
                    K_γsγsαβ[i,j] += (mp.lα^2 * ∇ϕ(i) ⋅ mp.Hgrad[β] ⋅ ∇ϕ(j) + ϕ(i) * Aτγ[β] * ϕ(j)) * detJdV(fev, q_point)
                end
            end
        end
    end

    f[u_dofs] = full(f_u)
    K[up◫, up◫] = K_uu
    for α in 1:nslip
        f[γ_dofs[α]] = f_γs[α]
        K[Block(Int(u◫) + α, 1)] = K_uγs[α]'
        K[Block(1, Int(u◫) + α)] = K_γsu[α]
        for β in 1:nslip
             K[Block(Int(u◫) + β, Int(u◫) + α)] = K_γsγs[α, β]
        end
    end

    end #inbounds
    return full(f), full(K)

end

function compute_tau(γ_gp, γ_gp_prev, ∆t, mp::CrystPlastMP)
    @unpack mp: C, tstar, n
    Δγ = γ_gp - γ_gp_prev
    τ = C * (tstar / ∆t * abs(Δγ))^(1/n)
    return sign(Δγ) * τ
end

function diff_tau(γ_gp, γ_gp_prev, ∆t, mp::CrystPlastMP)
    @unpack mp: C, tstar, n
    Δγ = γ_gp - γ_gp_prev
    C * (tstar / ∆t)^(1/n) * 1/n * abs(Δγ)^(1/n - 1)
end


