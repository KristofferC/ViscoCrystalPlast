function intf{dim, T, QD <: CrystPlastDualQD}(dual_prob::DualProblem,
                        a::Vector{T}, prev_a::Vector{T}, ɛ_bar, σ_bar, x::Vector, fev_u::CellVectorValues{dim}, fev_ξ::CellScalarValues{dim}, dt,
                        mss::AbstractVector{QD}, temp_mss::AbstractVector{QD}, mp, compute_stiffness::Bool)
    @unpack s, m, H⟂, Ee, sxm_sym, l = mp
    nslip = length(sxm_sym)

    glob_prob = dual_prob.global_problem

    @unpack u_nodes, ξo_nodes, ξ⟂_nodes = glob_prob
    @unpack u_dofs, ξ⟂s_dofs, ξos_dofs = glob_prob
    @unpack χ⟂, χo, Aγεs, δε, DAγξ⟂s, DAγξos = glob_prob
    @unpack Ω, problem_type = glob_prob

    if problem_type == Dirichlet
        Ω = 1.0
    end

    σv = fromvoigt(Tensor{2, dim}, σ_bar)

    @unpack f, C_f, K, C_K = glob_prob

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
    fill!(C_K, 0.0)
    fill!(f, 0.0)
    fill!(C_f, 0.0)

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
        ε = function_symmetric_gradient(fev_u, q_point, u_vec)
        ∇u_qp = function_gradient(fev_u, q_point, u_vec)
        for i in 1:nbasefuncs_u
            δε[i] = symmetric(shape_gradient(fev_u, q_point, i))
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
            X = solve_local_problem(Y, dual_prob.local_problem, dt, mp, ms, temp_ms)
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


        χ_π = 0.0
        ϕ_π = 0.0
        τ_π = 0.0
        ψg = 0.0
        ψe = 1/2 * ε_e ⊡ mp.Ee ⊡ ε_e
        if T == Float64
            temp_ms.σ  = σ
            temp_ms.ε  = ε
            temp_ms.ε_p = ε_p
            for α in 1:nslip
                ξ⟂ = function_value(fev_ξ, q_point, ξ⟂_nodes[α])
                ξo = function_value(fev_ξ, q_point, ξo_nodes[α])
                τ = (σ ⊡ mp.sxm_sym[α])
                Δγ = γ[α] - ms.γ[α]
                ψg += (1/2 * 1 / mp.lα^2) * (1/mp.H⟂ * ξ⟂^2 + 1/mp.Ho * ξo^2)
                τ_π += τ_di[α] * Δγ
                χ_π += (χ⟂[α] + χo[α]) * γ[α]
                ϕ_π += dt * 1/mp.tstar * mp.C/(mp.n+1) * (abs(τ)/mp.C)^(mp.n+1)

                temp_ms.χ⟂[α] = χ⟂[α]
                temp_ms.χo[α] = χo[α]
                temp_ms.τ_di[α] = τ_di[α]
                temp_ms.γ[α] = γ[α]
                temp_ms.τ[α] = τ
            end

            temp_ms.ψek = ms.ψe
            temp_ms.ψgk = ms.ψg

            φ = ψe - ψg
            φk = ms.ψe - ms.ψg

            temp_ms.ψe = ψe
            temp_ms.ψg = ψg
            temp_ms.χ_π = χ_π
            temp_ms.ϕ_π = ϕ_π
            temp_ms.τ_π = τ_π
            temp_ms.π = φ - φk + τ_π - χ_π - ϕ_π

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

        # C_f
        C_f += tovoigt(-1/Ω * (∇u_qp - ɛ_bar) * dΩ)

        for i in 1:nbasefuncs_u
            δ∇ui = shape_gradient(fev_u, q_point, i)

            # f_u
            if problem_type == Dirichlet
                f_u[i] += (δε[i] ⊡ σ) * dΩ
            else
                f_u[i] += 1/Ω * (δε[i] ⊡ σ - σv ⊡ δ∇ui) * dΩ
            end

            # C_K
            C_K[i, :] .+= tovoigt(-1/Ω * δ∇ui * dΩ)

            # K_uu
            if compute_stiffness
                δεEe_DA = δε[i] ⊡ Ee_DA
                for j in 1:nbasefuncs_u
                    K_uu[i, j] += 1/Ω * δεEe_DA ⊡ δε[j] * dΩ
                end

                # K_uξ
                for β in 1:nslip
                    β_offset = nbasefuncs_ξ * (β - 1)
                    δεDAγξ⟂sΒ = δε[i] ⊡ DAγξ⟂s[β]
                    for j in 1:nbasefuncs_ξ
                        ∇δξj = shape_gradient(fev_ξ, q_point, j)
                            K_uξ⟂s[i, β_offset + j] += - 1/Ω * (δεDAγξ⟂sΒ) * (∇δξj ⋅ mp.s[β]) * dΩ
                        if dim == 3
                            K_uξos[i, β_offset + j] += - 1/Ω * (δεDAγξ⟂sΒ) * (∇δξj ⋅ mp.l[β]) * dΩ
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
                δξi = shape_value(fev_ξ, q_point, i)
                ∇δξi = shape_gradient(fev_ξ, q_point, i)
                ∇δξ_i_s_α = ∇δξi ⋅ mp.s[α]
                ∇δξ_i_l_α = ∇δξi ⋅ mp.l[α]
                # f_ξ
                    f_ξ⟂s[α_offset + i] += - 1/Ω * (g⟂_gp * δξi + ∇δξ_i_s_α * γ[α]) * dΩ
                if dim == 3
                    f_ξos[α_offset + i] += - 1/Ω * (go_gp * δξi + ∇δξ_i_l_α * γ[α]) * dΩ
                end

                # K_ξu
                if compute_stiffness
                    ∇δξAγεs_⟂ = ∇δξ_i_s_α * Aγεs[α]
                    ∇δξAγεs_o = ∇δξ_i_l_α * Aγεs[α]
                    for j in 1:nbasefuncs_u
                            K_ξ⟂su[α_offset + i, j] += - 1/Ω * (∇δξAγεs_⟂ ⊡ δε[j]) * dΩ
                        if dim == 3
                            K_ξosu[α_offset + i, j] += - 1/Ω * (∇δξAγεs_o ⊡ δε[j]) * dΩ
                        end
                    end

                    for β in 1:nslip
                        # K_ξξ diag
                        β_offset = nbasefuncs_ξ * (β - 1)
                        for j in 1:nbasefuncs_ξ
                            δξj = shape_value(fev_ξ, q_point, j)
                            ∇δξj = shape_gradient(fev_ξ, q_point, j)
                            ∇δξ_j_s_β = ∇δξj ⋅ mp.s[β]
                            ∇δξ_j_l_β = ∇δξj ⋅ mp.l[β]
                            if α == β
                                K_ξ⟂sξ⟂s[α_offset + i, β_offset + j] += - 1/Ω * δξi  / (mp.H⟂ * mp.lα^2) * δξj * dΩ
                            end
                            K_ξ⟂sξ⟂s[α_offset + i, β_offset + j] += - 1/Ω * ∇δξ_i_s_α * Aγξ⟂[α, β] * ∇δξ_j_s_β * dΩ

                            if dim == 3
                                if α == β
                                    K_ξosξos[α_offset + i, β_offset + j] += - 1/Ω * δξi / (mp.Ho * mp.lα^2) * δξj * dΩ
                                end
                                K_ξosξos[α_offset + i, β_offset + j] += - 1/Ω * ∇δξ_i_l_α * Aγξo[α, β] * ∇δξ_j_l_β * dΩ
                                K_ξ⟂sξos[α_offset + i, β_offset + j] += - 1/Ω * ∇δξ_i_s_α * Aγξo[α, β] * ∇δξ_j_l_β * dΩ
                                K_ξosξ⟂s[α_offset + i, β_offset + j] += - 1/Ω * ∇δξ_i_l_α * Aγξ⟂[α, β] * ∇δξ_j_s_β * dΩ
                            end
                        end
                    end # compute_stiffness
                end
            end
        end
    end

    end # inbounds

    return f, C_f, K, C_K
end
