function intf{dim, T, QD <: CrystPlastPrimalQD}(primal_prob::PrimalProblem,
                           a::Vector{T}, a_prev::Vector{T}, x::Vector, fev_u::CellVectorValues{dim}, fev_γ::CellScalarValues{dim},
                          dt, mss::AbstractVector{QD}, temp_mss::AbstractVector{QD}, mp::CrystPlastMP, compute_stiffness::Bool)
    @unpack s, m, l, H⟂, Ho, Ee, sxm_sym = mp
    nslip = length(sxm_sym)

    glob_prob = primal_prob.global_problem

    @unpack u_nodes, γ_nodes, γ_prev_nodes = glob_prob
    @unpack u_dofs, γ_dofs = glob_prob
    @unpack Aτγ, δε = glob_prob
    @unpack f, K = glob_prob

    nbasefuncs_u = getnbasefunctions(fev_u)
    nbasefuncs_γ = getnbasefunctions(fev_γ)
    nnodes = getnbasefunctions(fev_γ)


    u_offset = nbasefuncs_u
    γ_offset = nbasefuncs_γ * nslip

    f_u      = @view f[1:u_offset]
    f_γs     = @view f[u_offset + 1:u_offset + γ_offset]

    K_uu     = @view K[1:u_offset, 1:u_offset]
    K_uγs    = @view K[1:u_offset, u_offset + 1:u_offset + γ_offset]

    K_γsu    = @view K[u_offset + 1:u_offset + γ_offset, 1:u_offset]
    K_γsγs   = @view K[u_offset + 1:u_offset + γ_offset, u_offset + 1:u_offset + γ_offset]

    @assert length(a) == length(a_prev) == nbasefuncs_u + nbasefuncs_γ * nslip

    reinit!(fev_u, x)
    reinit!(fev_γ, x)
    fill!(K, 0.0)
    fill!(f, 0.0)

    extract!(u_nodes, a, u_dofs)
    u_vec = reinterpret(Vec{dim, T}, u_nodes, (nnodes,))

    for α in 1:nslip
        extract!(γ_nodes[α], a, γ_dofs[α])
        extract!(γ_prev_nodes[α], a_prev, γ_dofs[α])
    end

    @inbounds begin

    for q_point in 1:getnquadpoints(fev_u)
        dΩ = getdetJdV(fev_u, q_point)
  
        ε = function_symmetric_gradient(fev_u, q_point, u_vec)
        for i in 1:nbasefuncs_u
            ∇δui = shape_gradient(fev_u, q_point, i)
            δε[i] = symmetric(∇δui)
        end

        ε_p = zero(ε)
        for α in 1:nslip
            γ = function_value(fev_γ, q_point, γ_nodes[α])
            ε_p += γ * sxm_sym[α]
        end

        ε_e = ε - ε_p
        σ = Ee ⊡ ε_e

        if T == Float64
            temp_mss[q_point].σ  = σ
            temp_mss[q_point].ε  = ε
            temp_mss[q_point].ε_p = ε_p
        end

        for i in 1:nbasefuncs_u
            # f_u
            f_u[i] += (δε[i] ⊡ σ) * dΩ
            if compute_stiffness
                # K_uu
                δεeE = δε[i] ⊡ Ee
                for j in 1:nbasefuncs_u
                    K_uu[i, j] += δεeE ⊡ δε[j] * dΩ
                end
                # K_uγ
                for β in 1:nslip
                    β_offset = nbasefuncs_γ * (β - 1)
                    δεEsm = δε[i] ⊡ mp.Esm[β]
                    for j in 1:nbasefuncs_γ
                        δγj = shape_value(fev_γ, q_point, j)
                        K_uγs[i, β_offset + j] += - δεEsm * δγj * dΩ
                    end # j
                end # β
            end # stiffness
        end # i

        for α in 1:nslip
            γ      = function_value(fev_γ, q_point, γ_nodes[α])
            γ_prev = function_value(fev_γ, q_point, γ_prev_nodes[α])
            τα = compute_tau(γ, γ_prev, dt, mp)
            τ_en = -(σ ⊡ sxm_sym[α])
            Aτγ[α] = diff_tau(γ, γ_prev, dt, mp)

            g = function_gradient(fev_γ, q_point, γ_nodes[α])
            ξ = mp.lα^2 * mp.Hgrad[α] ⋅ g
            α_offset = nbasefuncs_γ * (α - 1)
            # f_γ
            for i in 1:nbasefuncs_γ
                δγi = shape_value(fev_γ, q_point, i)
                ∇δγi = shape_gradient(fev_γ, q_point, i)
                f_γs[i + α_offset] += (δγi * (τα + τ_en) + ∇δγi ⋅ ξ) * dΩ

                for j in 1:nbasefuncs_u
                    K_γsu[i + α_offset, j] += - δγi * (mp.Esm[α] ⊡ δε[j]) * dΩ
                end
                # K_γu
                for β in 1:nslip
                    β_offset = nbasefuncs_γ * (β - 1)
                    ∇δγHgra = (∇δγi ⋅ mp.Hgrad[β])
                    for j in 1:nbasefuncs_γ
                        δγj = shape_value(fev_γ, q_point, j)
                        ∇δγj = shape_gradient(fev_γ, q_point, j)

                        K_γsγs[i + α_offset, j + β_offset] += (δγi * mp.Dαβ[α, β] * δγj) * dΩ
                        if α == β
                            K_γsγs[i + α_offset, j + β_offset] += (mp.lα^2 * (∇δγHgra ⋅ ∇δγj) + δγi * Aτγ[β] * δγj) * dΩ
                        end
                    end # j
                end # β
            end # i

            if T == Float64
                temp_mss[q_point].ξ⟂[α] = g ⋅ mp.s[α] * mp.lα^2 * mp.H⟂
                temp_mss[q_point].ξo[α] = g ⋅ mp.l[α] * mp.lα^2 * mp.Ho
                temp_mss[q_point].τ_di[α] = τα
                temp_mss[q_point].τ[α] = -τ_en
                temp_mss[q_point].γ[α] = γ
            end
        end # α
    end # qpoint

    end #inbounds
    return f, K
end

function compute_tau(γ_gp, γ_gp_prev, ∆t, mp::CrystPlastMP)
    @unpack C, tstar, n = mp
    Δγ = γ_gp - γ_gp_prev
    τ = C * (tstar / ∆t * abs(Δγ))^(1/n)
    return sign(Δγ) * τ
end

function diff_tau(γ_gp, γ_gp_prev, ∆t, mp::CrystPlastMP)
    @unpack C, tstar, n = mp
    Δγ = γ_gp - γ_gp_prev
    C * (tstar / ∆t)^(1/n) * 1/n * abs(Δγ)^(1/n - 1)
end
