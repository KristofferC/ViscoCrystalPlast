function update_problem!{dim}(problem::DualLocalProblem{dim}, Δt, mp, ms)
    @unpack problem: γ, τ, ε, χ⟂, χo, J_ττ, J_τγ, J_γτ, J_γγ, J, R_τ, R_γ, R, outer, inner

    nslip = length(mp.angles)

    @inbounds begin

    getblock!(ε, outer, ε◫)
    getblock!(χ⟂, outer, χ⟂◫)
    if dim == 3
        getblock!(χo, outer, χo◫)
    end
    εt = symmetric(Tensor{2, dim}(ε))

    getblock!(τ, inner, τ◫)
    getblock!(γ, inner, γ◫)

    ε_p = zero(εt)
    for α in 1:length(γ)
        ε_p += γ[α] * mp.sxm_sym[α]
    end

    ε_e = εt - ε_p
    σ = mp.Ee ⊡ ε_e

    for α in 1:nslip
        # Residual
        τen = -(σ ⊡ mp.sxm_sym[α])
        Δγ = γ[α] - ms.γ[α]
        R_γ[α] = τen + τ[α] - χ⟂[α]
        if dim == 3
            R_γ[α] -= χo[α]
        end
        R_τ[α] = Δγ - Δt / mp.tstar * (abs(τ[α]) / mp.C )^(mp.n) * sign(τ[α])

        # Jacobian
        for β in 1:nslip
            J_γγ[α, β] = mp.Dαβ[α, β]
            if α == β
                J_τγ[α, β] = 1.0
                J_ττ[α, β] = - Δt / mp.tstar * mp.n / mp.C * (abs(τ[α]) / mp.C )^(mp.n-1)
                J_γτ[α, β] = 1.0
            end
        end
    end

    R[γ◫] = R_γ
    R[τ◫] = R_τ

    J[γ◫, τ◫] = J_γτ
    J[γ◫, γ◫] = J_γγ
    J[τ◫, τ◫] = J_ττ
    J[τ◫, γ◫] = J_τγ

    end # inbounds

    return
end

function update_ats!{dim}(problem::DualLocalProblem{dim}, Δt, mp, ms)
    @unpack problem: γ, τ, ε, χ⟂, χo, Q_γε, Q_γχ⟂, Q_γχo, Q_τε, Q_τχ⟂, Q_τχo, Q
    nslip = length(mp.angles)

    @inbounds begin
    for α in 1:nslip
        Q_γε[α, :] = -mp.Esm[α]

        Q_γχ⟂[α, α] = -1.0
        if dim == 3
            Q_γχo[α, α] = -1.0
        end
    end

    Q[γ◫, ε◫] = Q_γε
    Q[γ◫, χ⟂◫] = Q_γχ⟂
    if dim == 3
        Q[γ◫, χo◫] = Q_γχo
    end
    end # inbounds
end

function consistent_tangent{dim}(out::Vector, problem::DualLocalProblem{dim}, ∆t, mp, ms, temp_ms)
    # The problem here is assumed to be solved and the results are in the gauss point data in temp_ms.
    reset!(problem)
    problem.inner[τ◫] = temp_ms.τ_di
    problem.inner[γ◫] = temp_ms.γ
    copy!(full(problem.outer), out)

    update_problem!(problem, ∆t, mp, ms)
    @assert norm(problem.R) <= 1e-6
    update_ats!(problem, ∆t, mp, ms)
    A = full(problem.J) \ full(problem.Q)
    scale!(A, -1)
    copy!(problem.A, A)

    return return problem.A
end


function solve_local_problem{T}(out::Vector{T}, problem::DualLocalProblem, ∆t, mp, ms, temp_ms)
    # Set initial value

    @inbounds problem.inner[γ◫] = ms.γ
    @inbounds problem.inner[τ◫] = ms.τ_di

    copy!(full(problem.outer), out)


    newton_solve!(out, problem, ∆t, mp, ms, temp_ms)


    return full(problem.inner)
end

function newton_solve!(out, problem, ∆t, mp, ms, temp_ms)
    max_iters = 40
    n_iters = 1

    while true
        reset!(problem)
        update_problem!(problem, ∆t, mp, ms)
        res = norm(full(problem.R), Inf)

        if res  <= 1e-7
            break
        end

        A_ldiv_B!(lufact!(full(problem.J)), full(problem.R))
        @inbounds for i in eachindex(full(problem.inner))
            problem.inner[i] -= problem.R[i]
        end
        if n_iters == max_iters
            error("Non conv mat")
        end
        n_iters +=1
    end
end

#=
function foo()
    srand(1234)
    nslip = 2
    dim = 2
    a = rand(dim == 2? 4 : 9);
    b = rand(nslip);
    c = rand(nslip);
    d =  rand(nslip);

    problem = ViscoCrystalPlast.DualLocalProblem(nslip, Dim{dim});
    out = [a; b] #rand(dim == 2? 4 : 9);
    problem.inner[γ◫] = c; #rand(nslip);
    problem.inner[τ◫] = d; #rand(nslip);

    mp = setup_material(Dim{dim});
    ms = ViscoCrystalPlast.CrystPlastDualQD(nslip, Dim{dim});
    temp_ms = ViscoCrystalPlast.CrystPlastDualQD(nslip, Dim{dim});


    X = ViscoCrystalPlast.solve_local_problem(out, problem, 0.1, mp, ms, temp_ms)
    #copy!(full(problem.inner), X)
    temp_ms.γ = problem.inner[γ◫]
    temp_ms.τ_di = problem.inner[τ◫]
    @time for i in 1:10^5
        ViscoCrystalPlast.solve_local_problem(out, problem, 0.1, mp, ms, temp_ms)
    end


    @time for i in 1:10^5
        update_problem!(problem, 0.1, mp, ms)
        consistent_tangent(out, 0.1, mp, temp_ms, problem)
    end


end
=#