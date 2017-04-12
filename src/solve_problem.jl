function solve_problem{dim}(problem::AbstractProblem, mesh::Grid, dofhandler::DofHandler, dbcs::DirichletBoundaryConditions,
                            fev_u::CellVectorValues{dim}, fev_ξ::CellScalarValues{dim}, mps::Vector, timesteps, exporter, polys, ɛ_bar_f)

    @timeit "sparsity pattern" begin
        K = create_sparsity_pattern(dofhandler)
    end
    total_dofs = size(K,1)
    n_dofs_u = dofhandler.ndofs[1]
    n_dofs_ξ = dofhandler.ndofs[2]
    @assert n_dofs_u + n_dofs_ξ == total_dofs

    println("Analysis started\n----------------")
    println("Time steps: ", length(timesteps), " steps [", first(timesteps), ",", last(timesteps), "]")
    println("Mesh: n_elements: ", getncells(mesh), ", n_nodes:", getnnodes(mesh))
    println("Dofs: n_dofs: ", total_dofs)

    free = dbcs.free_dofs
    ∆u = 0.0001 * rand(total_dofs)
    u = zeros(∆u)
    ∆∆σ_bar = zeros(dim*dim)
    σ_bar = zeros(∆∆σ_bar)
    σ_bar_n = zeros(∆∆σ_bar)
    apply_zero!(∆u, dbcs)
    un = zeros(total_dofs)

    # Guess

    full_residuals = zeros(total_dofs)
    n_qpoints = getnquadpoints(fev_u)

    nslip = length(mps[1].s)

#    if isa(problem, PrimalProblem)
        #mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:getncells(mesh)]
        #temp_mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:getncells(mesh)]
    #else
        mss = [CrystPlastDualQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:getncells(mesh)]
        temp_mss = [CrystPlastDualQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:getncells(mesh)]
    #end

    #ps = MKLPardisoSolver()
    #set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
    #pardisoinit(ps)
    use_Neumann = true
    t_prev = timesteps[1]
    ɛ_bar_p = ɛ_bar_f(first(timesteps))
    first_fact = true
    for (nstep, t) in enumerate(timesteps[2:end])
        ɛ_bar = ɛ_bar_f(t)
        ∆σ_bar = tovoigt(convert(Tensor{2,dim}, mps[1].Ee ⊡ (ɛ_bar - ɛ_bar_p)))
        ɛ_bar_p = ɛ_bar
        dt = t - t_prev
        t_prev = t
        println("Step $nstep, t = $t, dt = $dt")

        JuAFEM.update!(dbcs, t)
        apply!(un, dbcs) # Make primary field obey BC

        iter = 1
        max_iters = 10

        while iter == 1 || (norm(f[free]./f_sq[free], Inf)  >= 1e-5 && norm(f[free], Inf) >= 1e-8) ||
                            (problem.global_problem.problem_type == Neumann && use_Neumann && norm(C ./ C_sq, Inf)  >= 1e-5 && norm(C, Inf) > 1e-6)
            u .= un .+ ∆u
            σ_bar .= σ_bar_n .+ ∆σ_bar
            @timeit "assemble" begin
                K, C_K, f, f_sq, C, C_sq = assemble!(problem, K, u, un, ɛ_bar, σ_bar, fev_u, fev_ξ,
                                       mesh, dofhandler, dbcs, mps, mss, temp_mss, dt, polys)
            end



            if use_Neumann && problem.global_problem.problem_type == Neumann
                KK = [K      C_K
                     C_K' zeros(9,9)]
                ff = [f; C]
            else
              KK = K
              ff = f
            end

            if first_fact == true
              first_fact = false
              K_pardiso = get_matrix(ps, KK, :N)
              set_phase!(ps, Pardiso.ANALYSIS)
              pardiso(ps, K_pardiso, z)
            end


            if iter > max_iters
                error("Newton iterations did not converge")
            end

            @show norm(f[free]./f_sq[free], Inf)
            @show norm(f[free], Inf)
            @show norm(C ./ C_sq, Inf)
            println("----")

            full_residuals[free] = f[free]

            println("Step: $nstep, iter: $iter")
            print("Error: ")
            print_residuals(dofhandler, full_residuals)

            apply_zero!(KK, ff, dbcs)
            apply_zero!(K, f, dbcs)
            @timeit "factorization" begin
                #=
                @timeit "pardiso" begin
                  K_pardiso = get_matrix(ps, KK, :N)
                  set_phase!(ps, Pardiso.NUM_FACT)
                  pardiso(ps, K_pardiso, ff)
                  set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
                  X = similar(ff) # Solution is stored in X
                  pardiso(ps, X, K_pardiso, ff)
                end

                @timeit "MUMPS"  begin
                  solveMUMPS(KK, ff, 1);
                end
                =#

              #  @timeit "SUITSPARSE" begin
              #  @show "solving sparse"
                  ∆∆u = KK \ ff #
            #    end
            end

            if use_Neumann && problem.global_problem.problem_type == Neumann
                ∆∆σ_bar = ∆∆u[end-8:end]
                ∆∆u = ∆∆u[1:end-9]
                ∆σ_bar .-= ∆∆σ_bar
                @assert length(∆σ_bar) == 9
            end

            @show C

            @show length(∆∆u)
            @show ndofs(dbcs.dh)

            apply_zero!(∆∆u, dbcs)
            ∆u .-= ∆∆u
            iter += 1
        end

        copy!(un, u)
        copy!(σ_bar_n, σ_bar)

        # temp_mss is now the true matstats
        mss, temp_mss = temp_mss, mss
        exporter(t, u, mss)
    end
    return u, σ_bar_n, mss
end

using Calculus

function assemble!{dim}(problem, K::SparseMatrixCSC, u::Vector, un::Vector, ɛ_bar, σ_bar,
                        fev_u::CellVectorValues{dim}, fev_ξ::CellScalarValues{dim}, mesh::Grid, dofhandler::DofHandler,
                        bcs::DirichletBoundaryConditions, mps::Vector, mss, temp_mss, dt::Float64, polys::Vector{Int})

    #@assert length(u) == length(dofs.dof_ids)
    total_dofs = size(K,1)
    n_dofs_u = dofhandler.ndofs[1]

    global_dofs = zeros(Int, ndofs_per_cell(dofhandler))
    f_int = zeros(total_dofs)
    f_int_sq = zeros(total_dofs)
    C_int = zeros(9)
    C_int_sq = zeros(9)
    C_K = zeros(total_dofs, 9)
    # Internal force vector
    assembler = start_assemble(K, f_int)
    println("assembling...")
    @timeit "assemble loop" begin
        for element_id in 1:getncells(mesh)
            celldofs!(global_dofs, dofhandler, element_id)
            element_coords = getcoordinates(mesh, element_id)

            u_e = u[global_dofs]
            un_e = un[global_dofs]

            ele_matstats = view(mss, :, element_id)
            temp_matstats = view(temp_mss, :, element_id)

            fe(field) = intf(problem, field, un_e, ɛ_bar, σ_bar, element_coords,
                            fev_u, fev_ξ, dt, ele_matstats, temp_matstats, mps[polys[element_id]], true)

            #=
            function fee(uσ)
                uz = uσ[1:ndofs_per_cell(dofhandler)]
                σ = uσ[ndofs_per_cell(dofhandler)+1:end]

                fe_int, C_f, Ke, C_Ke = intf(problem, uz, un_e, ɛ_bar, σ, element_coords,
                                fev_u, fev_ξ, dt, ele_matstats, temp_matstats, mps[polys[element_id]], true)

                return [fe_int; C_f]
            end

            uσ = [u_e; σ_bar]
            Kee = Calculus.jacobian(fee, uσ, :central)

            Ke = Kee[1:ndofs_per_cell(dofhandler), 1:ndofs_per_cell(dofhandler)]
            C_Ke = Kee[1:ndofs_per_cell(dofhandler), ndofs_per_cell(dofhandler)+1:end]
            =#

            @timeit "intf" begin
                fe_int, C_f, Ke, C_Ke = fe(u_e)
            end

            #@show Ke
            #@show size(Kee)
            #@show size(Ke)
            #@show C_f
            #@show size(C_f)

            @timeit "assemble to global" begin
                JuAFEM.assemble!(assembler, fe_int, Ke, global_dofs)
                JuAFEM.assemble!(f_int_sq, fe_int.^2, global_dofs)
                C_int .+= C_f
                C_int_sq .+= C_f .* C_f
                for (i, dof) in enumerate(global_dofs[1:getnbasefunctions(fev_u)])
                    C_K[dof, :] += C_Ke[i, :]
                end
            end
        end
    end

    return K, C_K, f_int, f_int_sq, C_int, C_int_sq
end
