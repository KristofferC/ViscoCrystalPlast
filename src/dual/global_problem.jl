
function intf{dim, T, Q, MS <: CrystPlastDualQD}(
                        dual_prob::DualProblem{dim, T}, a::Vector{T}, prev_a::AbstractArray{Q},
                        x::AbstractArray{Q}, fev::FEValues{dim}, dt,
                        mss::AbstractVector{MS}, temp_mss::AbstractVector{MS}, mp::CrystPlastMP)
    @unpack mp: s, m, H⟂, Ee, sxm_sym, l
    nslip = length(sxm_sym)

    glob_prob = dual_prob.global_problem

    @unpack glob_prob: u_nodes, ξo_nodes, ξ⟂_nodes
    @unpack glob_prob: f_u , f_ξ⟂s, f_ξos, f
    @unpack glob_prob: K_uu, K_uξ⟂s, K_uξos, K_ξ⟂su, K_ξ⟂sξ⟂s, K_ξ⟂sξos, K_ξosu, K_ξosξ⟂s, K_ξosξos, K
    @unpack glob_prob: u_dofs, ξ⟂s_dofs, ξos_dofs
    @unpack glob_prob: χ⟂, χo, Aγεs

    ngradvars = dim - 1
    nnodes = n_basefunctions(get_functionspace(fev))

    @assert length(a) == nnodes * (dim + ngradvars * nslip)

    x_vec = reinterpret(Vec{dim, Q}, x, (nnodes,))
    reinit!(fev, x_vec)
    reset!(glob_prob)

    extract!(u_nodes, a, u_dofs)

    u_vec = reinterpret(Vec{dim, T}, u_nodes, (nnodes,))

    @inbounds begin

    for α in 1:nslip
        extract!(ξ⟂_nodes[α], a, ξ⟂s_dofs[α])
        if dim == 3
            extract!(ξo_nodes[α], a, ξos_dofs[α])
        end
    end

    for q_point in 1:length(JuAFEM.points(get_quadrule(fev)))
        ϕ = i -> shape_value(fev, q_point, i)
        ∇ϕ = i -> shape_gradient(fev, q_point, i)
        ε = function_vector_symmetric_gradient(fev, q_point, u_vec)

        for α in 1:nslip
            χ⟂[α] = function_scalar_gradient(fev, q_point, ξ⟂_nodes[α]) ⋅ mp.s[α]
            if dim == 3
                χo[α] = function_scalar_gradient(fev, q_point, ξo_nodes[α]) ⋅ mp.l[α]
            end
        end

        @assert nslip == 2
        if dim == 2
            Y = [ε[1,1], ε[2,1], ε[1,2], ε[2,2], χ⟂[1], χ⟂[2]]
        else
            Y = [ε[1,1], ε[2,1], ε[3,1], ε[1,2], ε[2,2], ε[3,2], ε[1,3], ε[2,3], ε[3,3],  χ⟂[1], χ⟂[2], χo[1], χo[2]]
        end
        ms = mss[q_point]
        temp_ms = temp_mss[q_point]


        X = solve_local_problem(Y, dual_prob.local_problem, dt, mp, ms, temp_ms)

        γ = X[γ◫]
        τ_di = X[τ◫]

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


        A = consistent_tangent(Y, dual_prob.local_problem, dt, mp, ms, temp_ms)

        Aγε = A[γ◫, ε◫]
        Aγξ⟂ = A[γ◫, ξ⟂◫]
        if dim == 3
            Aγξo = A[γ◫, ξo◫]
        end

        #######################
        # Displacement + grad #
        #######################
        DA = zero(SymmetricTensor{4, dim})
        for α in 1:nslip
            Aγεa = Aγε[α, :]
            DA += mp.Esm[α] ⊗ symmetric(Tensor{2, dim}(Aγεa))
            Aγεs[α] = Tensor{2, dim}(Aγεa)
        end

        DAγξ⟂s = [zero(SymmetricTensor{2, dim}) for α in 1:nslip]
        if dim == 3
            DAγξos = [zero(SymmetricTensor{2, dim}) for α in 1:nslip]
        end
        for α in 1:nslip, β in 1:nslip
            DAγξ⟂s[β] += mp.Esm[α] * Aγξ⟂[α, β]
            if dim == 3
                DAγξos[β] += mp.Esm[α] * Aγξo[α, β]
            end
        end

        for i in 1:nnodes
            ∇ϕ(i) = shape_gradient(fev, q_point, i)
            updateblock!(f_u, σ ⋅ ∇ϕ(i) * detJdV(fev, q_point), +, i)
            for j in 1:nnodes
                updateblock!(K_uu, dotdot(∇ϕ(i), Ee - DA, ∇ϕ(j)) * detJdV(fev, q_point), +, i, j)
                for β in 1:nslip
                    K_uξ⟂_qp = -(∇ϕ(i) ⋅ DAγξ⟂s[β]) * (∇ϕ(j) ⋅ mp.s[β]) * detJdV(fev, q_point)
                    updateblock!(K_uξ⟂s[β], K_uξ⟂_qp, +, i, j)

                    K_ξ⟂s_qp = -∇ϕ(i) ⋅ Aγεs[β] * (∇ϕ(j) ⋅ mp.s[β]) * detJdV(fev, q_point)
                    updateblock!(K_ξ⟂su[β], K_ξ⟂s_qp, +, i, j)

                    if dim == 3
                        K_uξo_qp = -(∇ϕ(i) ⋅ DAγξos[β]) * (∇ϕ(j) ⋅ mp.l[β]) * detJdV(fev, q_point)
                        updateblock!(K_uξos[β], K_uξo_qp, +, i, j)

                        K_ξos_qp = -∇ϕ(i) ⋅ Aγεs[β] * (∇ϕ(j) ⋅ mp.l[β]) * detJdV(fev, q_point)
                        updateblock!(K_ξosu[β], K_ξos_qp, +, i, j)
                    end
                end
            end
        end

        ###############
        # grad + grad #
        ###############
        for α in 1:nslip
            ξ⟂_gp = function_scalar_value(fev, q_point, ξ⟂_nodes[α])
            g⟂_gp = ξ⟂_gp / (mp.H⟂ * mp.lα^2)

            if dim == 3
                ξo_gp = function_scalar_value(fev, q_point, ξo_nodes[α])
                go_gp = ξo_gp / (mp.Ho * mp.lα^2)
            end

            for i in 1:nnodes
                f_ξ⟂s[α][i] += -(g⟂_gp * ϕ(i) +  ∇ϕ(i) ⋅ mp.s[α] * γ[α]) * detJdV(fev, q_point)
                if dim == 3
                    f_ξos[α][i] += -(go_gp * ϕ(i) + γ[α] * ∇ϕ(i) ⋅ mp.l[α]) * detJdV(fev, q_point)
                end
                for j in 1:nnodes
                    for β in 1:nslip
                        if α == β
                            K_ξ⟂sξ⟂s[α, β][i,j] += -ϕ(i) / (mp.H⟂ * mp.lα^2) * ϕ(j) * detJdV(fev, q_point)
                        end
                        K_ξ⟂sξ⟂s[α, β][i,j] += -∇ϕ(i) ⋅ mp.s[β] * Aγξ⟂[β, α] * ∇ϕ(j) ⋅ mp.s[α] * detJdV(fev, q_point)
                        if dim == 3
                            if α == β
                                K_ξosξos[α, β][i,j] += -ϕ(i) / (mp.Ho * mp.lα^2) * ϕ(j) * detJdV(fev, q_point)
                            end
                            K_ξosξos[α, β][i,j] += -∇ϕ(i) ⋅ mp.l[α] * Aγξo[β, α] * ∇ϕ(j) ⋅ mp.l[α] * detJdV(fev, q_point)

                            K_ξ⟂sξos[α, β][i,j] += -∇ϕ(i) ⋅ mp.s[α] * Aγξo[β, α] * ϕ(j) * detJdV(fev, q_point)
                            K_ξosξ⟂s[α, β][i,j] += -∇ϕ(i) ⋅ mp.l[α] * Aγξ⟂[β, α] * ϕ(j) * detJdV(fev, q_point)
                        end
                    end
                end
            end
        end
    end

    f[u◫] = full(f_u)
    for α in 1:nslip
        if dim == 2
            f[ξ⟂s_dofs[α]] = f_ξ⟂s[α]
        else
            f[ξ⟂s_dofs[α]] = f_ξ⟂s[α]
            f[ξos_dofs[α]] = f_ξos[α]
        end
    end
    K[u◫, u◫] = K_uu
    for α in 1:nslip
        K[Block(Int(u◫) + α, Int(u◫))] = K_uξ⟂s[α]'


        K[Block(Int(u◫), Int(u◫) + α)] = K_ξ⟂su[α]

        if dim == 3
            K[Block(Int(u◫) + nslip + α, Int(u◫))] = K_uξos[α]'
            K[Block(Int(u◫), Int(u◫) + α + nslip)] = K_ξosu[α]
        end
        for β in 1:nslip
            K[Block(Int(u◫) + β, Int(u◫) + α)] = K_ξ⟂sξ⟂s[α, β]

            if dim == 3
                K[Block(Int(u◫) + β + nslip, Int(u◫) + α + nslip)] = K_ξosξos[α, β]

                K[Block(Int(u◫) + α, Int(u◫) + β + nslip)] = K_ξ⟂sξos[α, β]
                K[Block(Int(u◫) + α + nslip, Int(u◫) + β)] = K_ξosξ⟂s[α, β]
            end
        end
    end
    end # inbounds
    return full(f), full(K)
end
