function intf{dim, T, QD <: CrystPlastDualQD}(dual_prob::DualProblem,
                        a::Vector{T}, prev_a::Vector{T}, x::Vector, fev_u::CellVectorValues{dim}, fev_ξ::CellScalarValues{dim}, dt,
                        mss::AbstractVector{QD}, temp_mss::AbstractVector{QD}, mp::CrystPlastMP, compute_stiffness::Bool)
    @unpack s, m, H⟂, Ee, sxm_sym, l = mp
    nslip = length(sxm_sym)

    glob_prob = dual_prob.global_problem

    @unpack u_nodes, ξo_nodes, ξ⟂_nodes = glob_prob
    @unpack u_dofs, ξ⟂s_dofs, ξos_dofs = glob_prob
    @unpack χ⟂, χo, Aγεs, δε, DAγξ⟂s, DAγξos = glob_prob

    @unpack f, K = glob_prob

    nbasefuncs_u = getnbasefunctions(fev_u)
    nbasefuncs_ξ = getnbasefunctions(fev_ξ)
    nnodes = getnbasefunctions(fev_ξ)

    u_offset = nbasefuncs_u #size(K_uu, 1)
    ξ_offset = nbasefuncs_ξ * nslip #size(K_uξ⟂s, 2)

    f_u      = @view f[1:u_offset]
    f_ξ⟂s    = @view f[u_offset + 1:u_offset + ξ_offset]

    K_uu     = @view K[1:u_offset, 1:u_offset]
    K_uξ⟂s   = @view K[1:u_offset, u_offset + 1:u_offset + ξ_offset]

    K_ξ⟂sξ⟂s = @view K[u_offset + 1:u_offset + ξ_offset, u_offset + 1:u_offset + ξ_offset]
    K_ξ⟂su   = @view K[u_offset + 1:u_offset + ξ_offset, 1:u_offset]

   if dim == 3
       f_ξos    = @view f[u_offset + ξ_offset + 1:u_offset + 2ξ_offset]
       K_uξos   = @view K[1:u_offset, u_offset + ξ_offset + 1:u_offset + 2ξ_offset]
       K_ξ⟂sξos = @view K[u_offset + 1:u_offset + ξ_offset, u_offset + ξ_offset + 1:u_offset + 2ξ_offset]
       K_ξosξos = @view K[u_offset + ξ_offset + 1:u_offset + 2ξ_offset, u_offset + ξ_offset + 1:u_offset + 2ξ_offset]
       K_ξosu   = @view K[u_offset + ξ_offset + 1:u_offset + 2ξ_offset, 1:u_offset]
       K_ξosξ⟂s = @view K[u_offset + ξ_offset + 1:u_offset + 2ξ_offset, u_offset + 1:u_offset + ξ_offset]
   end

    @assert length(a) == nbasefuncs_u + (dim - 1) * nbasefuncs_ξ * nslip

    reinit!(fev_u, x)
    reinit!(fev_ξ, x)
    fill!(K, 0.0)
    fill!(f, 0.0)

    extract!(u_nodes, a, u_dofs)
    u_vec = reinterpret(Vec{dim, T}, u_nodes, (nnodes,))

    @inbounds begin

    for α in 1:nslip
        extract!(ξ⟂_nodes[α], a, ξ⟂s_dofs[α])
        if dim == 3
            extract!(ξo_nodes[α], a, ξos_dofs[α])
        end
    end


    for q_point in 1:getnquadpoints(fev_u)
        dΩ = getdetJdV(fev_u, q_point)
        δu = (i) -> shape_value(fev_u, q_point, i)
        ∇δu = (i) -> shape_gradient(fev_u, q_point, i)

        δξ = (i) -> shape_value(fev_ξ, q_point, i)
        ∇δξ = (i) -> shape_gradient(fev_ξ, q_point, i)

        ε = function_symmetric_gradient(fev_u, q_point, u_vec)
        for i in 1:nbasefuncs_u
            δε[i] = symmetric(∇δu(i))
        end

        for α in 1:nslip
            χ⟂[α] = function_gradient(fev_ξ, q_point, ξ⟂_nodes[α]) ⋅ mp.s[α]
            if dim == 3
                χo[α] = function_gradient(fev_ξ, q_point, ξo_nodes[α]) ⋅ mp.l[α]
            end
        end

        if dim == 2
            Y = vcat(vec(ε), χ⟂) # Symmetric?
        else
            Y = vcat(vec(ε), χ⟂, χo)
        end

        temp_ms = temp_mss[q_point]
        ms = mss[q_point]
        @timeit "local problem" begin
            X = solve_local_problem(Y, dual_prob.local_problem, dt, mp, ms, temp_ms )
        end

        γ = X[γ◫]
        τ_di = X[τ◫]

        # Store
        ε_p = zero(ε)
        for α in 1:nslip
            ε_p += γ[α] * sxm_sym[α]
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

        @timeit "consistent tangent" begin
            A = consistent_tangent(Y, dual_prob.local_problem, dt, mp, ms, temp_ms)
        end

        Aγε = A[γ◫, ε◫]
        Aγξ⟂ = A[γ◫, ξ⟂◫]
        if dim == 3
            Aγξo = A[γ◫, ξo◫]
        end

        DA = zero(SymmetricTensor{4, dim})
        for α in 1:nslip
            Aγεa = Aγε[α, :]
            Aγεat = symmetric(Tensor{2, dim}(Aγεa))
            DA += mp.Esm[α] ⊗ Aγεat
            Aγεs[α] = Aγεat
        end

        fill!(DAγξ⟂s, zero(SymmetricTensor{2, dim}))
        if dim == 3
            fill!(DAγξos, zero(SymmetricTensor{2, dim}))
        end
        for α in 1:nslip, β in 1:nslip
            DAγξ⟂s[β] += mp.Esm[α] * Aγξ⟂[α, β]
            if dim == 3
                DAγξos[β] += mp.Esm[α] * Aγξo[α, β]
            end
        end

        Ee_DA = Ee - DA

        for i in 1:nbasefuncs_u
            # f_u
            f_u[i] = (δε[i] ⊡ σ) * dΩ

            # K_uu
            if compute_stiffness
                δεEe_DA = δε[i] ⊡ Ee_DA
                for j in 1:nbasefuncs_u
                    K_uu[i, j] += δεEe_DA ⊡ δε[j] * dΩ
                end

                # K_uξ
                for β in 1:nslip
                    β_offset = nbasefuncs_ξ * (β - 1)
                    δεDAγξ⟂sΒ = δε[i] ⊡ DAγξ⟂s[β]
                    for j in 1:nbasefuncs_ξ
                            K_uξ⟂s[i, β_offset + j] += -(δεDAγξ⟂sΒ) * (∇δξ(j) ⋅ mp.s[β]) * dΩ
                        if dim == 3
                            K_uξos[i, β_offset + j] += -(δεDAγξ⟂sΒ) * (∇δξ(j) ⋅ mp.l[β]) * dΩ
                        end
                    end
                end
            end
        end

        for α in 1:nslip
            ξ⟂_gp = function_value(fev_ξ, q_point, ξ⟂_nodes[α])
            g⟂_gp = ξ⟂_gp / (mp.H⟂ * mp.lα^2)

            if dim == 3
                ξo_gp = function_value(fev_ξ, q_point, ξo_nodes[α])
                go_gp = ξo_gp / (mp.Ho * mp.lα^2)
            end

            α_offset = nbasefuncs_ξ * (α - 1)
            for i in 1:nbasefuncs_ξ
                ∇δξ_i_s_α = ∇δξ(i) ⋅ mp.s[α]
                ∇δξ_i_l_α = ∇δξ(i) ⋅ mp.l[α]

                # f_ξ
                    f_ξ⟂s[α_offset + i] += -(g⟂_gp * δξ(i) + ∇δξ_i_s_α  * γ[α]) * dΩ
                if dim == 3
                    f_ξos[α_offset + i] += -(go_gp * δξ(i) + ∇δξ_i_l_α * γ[α]) * dΩ
                end

                # K_ξu
                if compute_stiffness
                    ∇δξAγεs_⟂ = ∇δξ_i_s_α * Aγεs[α]
                    ∇δξAγεs_o = ∇δξ_i_l_α * Aγεs[α]
                    for j in 1:nbasefuncs_u
                            K_ξ⟂su[α_offset + i, j] += - (∇δξAγεs_⟂ ⊡ δε[j]) * dΩ
                        if dim == 3
                            K_ξosu[α_offset + i, j] += - (∇δξAγεs_o ⊡ δε[j]) * dΩ
                        end
                    end


                    for β in 1:nslip
                        # K_ξξ diag
                        β_offset = nbasefuncs_ξ * (β - 1)
                        for j in 1:nbasefuncs_ξ
                            ∇δξ_j_s_β = ∇δξ(j) ⋅ mp.s[β]
                            ∇δξ_j_l_β = ∇δξ(j) ⋅ mp.l[β]
                            if α == β
                                c = -δξ(i) / (mp.H⟂ * mp.lα^2) * δξ(j) * dΩ

                                K_ξ⟂sξ⟂s[α_offset + i, β_offset + j] += -δξ(i) / (mp.H⟂ * mp.lα^2) * δξ(j) * dΩ
                            end
                            K_ξ⟂sξ⟂s[α_offset + i, β_offset + j] += - ∇δξ_i_s_α * Aγξ⟂[α, β] * ∇δξ_j_s_β * dΩ

                            if dim == 3
                                if α == β
                                    K_ξosξos[α_offset + i, β_offset + j] += -δξ(i) / (mp.Ho * mp.lα^2) * δξ(j) * dΩ
                                end
                                K_ξosξos[α_offset + i, β_offset + j] += -∇δξ_i_l_α * Aγξo[α, β] * ∇δξ_j_l_β * dΩ
                                K_ξ⟂sξos[α_offset + i, β_offset + j] += -∇δξ_i_s_α * Aγξo[α, β] * ∇δξ_j_l_β * dΩ
                                K_ξosξ⟂s[α_offset + i, β_offset + j] += -∇δξ_i_l_α * Aγξ⟂[α, β] * ∇δξ_j_s_β * dΩ
                            end
                        end
                    end # compute_stiffness
                end
            end
        end
    end

    end # inbounds

    return f, K
end
