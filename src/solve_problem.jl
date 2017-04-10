function solve_problem{dim}(problem::AbstractProblem, mesh::Grid, dofhandler::DofHandler, dbcs::DirichletBoundaryConditions,
                            fev_u::CellVectorValues{dim}, fev_ξ::CellScalarValues{dim}, mps::Vector, timesteps, exporter, polys)

    @timeit "sparsity pattern" begin
        K = create_sparsity_pattern(dofhandler)
    end
    total_dofs = size(K,1)

    println("Analysis started\n----------------")
    println("Time steps: ", length(timesteps), " steps [", first(timesteps), ",", last(timesteps), "]")
    println("Mesh: n_elements: ", getncells(mesh), ", n_nodes:", getnnodes(mesh))
    println("Dofs: n_dofs: ", total_dofs)

    free = dbcs.free_dofs
    ∆u = 0.0001 * rand(total_dofs)
    apply_zero!(∆u, dbcs)
    u = zeros(total_dofs)
    un = copy(u)

    full_residuals = zeros(total_dofs)
    n_qpoints = getnquadpoints(fev_u)

    nslip = length(mps[1].s)

    if isa(problem, PrimalProblem)
        mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:getncells(mesh)]
        temp_mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:getncells(mesh)]
    else
        mss = [CrystPlastDualQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:getncells(mesh)]
        temp_mss = [CrystPlastDualQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:getncells(mesh)]
    end

    t_prev = timesteps[1]
    for (nstep, t) in enumerate(timesteps[2:end])
        dt = t - t_prev
        t_prev = t
        println("Step $nstep, t = $t, dt = $dt")

        JuAFEM.update!(dbcs, t)
        apply!(un, dbcs) # Make primary field obey BC

        iter = 1
        max_iters = 40

        while iter == 1 || norm(f[free]./f_sq[free], Inf)  >= 1e-5 && norm(f[free], Inf) >= 1e-8
            u .= un .+ ∆u
            @timeit "assemble" begin
                K, f, f_sq = assemble!(problem, K, u, un, fev_u, fev_ξ,
                                       mesh, dofhandler, dbcs, mps, mss, temp_mss, dt, polys)
            end

            if iter > max_iters
                error("Newton iterations did not converge")
            end

            full_residuals[free] = f[free]

            println("Step: $nstep, iter: $iter")
            print("Error: ")
            print_residuals(dofhandler, full_residuals)

            apply_zero!(K, f, dbcs)

            @timeit "factorization" begin
                ΔΔu = solveMUMPS(K, f, 1, 1);
            end
            apply_zero!(ΔΔu, dbcs)
            ∆u .-= ΔΔu
            iter += 1
        end

        copy!(un, u)

        # temp_mss is now the true matstats
        mss, temp_mss = temp_mss, mss
        exporter(t, u, mss)
    end
    return u, mss
end

function assemble!{dim}(problem, K::SparseMatrixCSC, u::Vector, un::Vector,
                        fev_u::CellVectorValues{dim}, fev_ξ::CellScalarValues{dim}, mesh::Grid, dofhandler::DofHandler,
                        bcs::DirichletBoundaryConditions, mps::Vector, mss, temp_mss, dt::Float64, polys::Vector{Int})

    #@assert length(u) == length(dofs.dof_ids)
    total_dofs = size(K,1)
    global_dofs = zeros(Int, ndofs_per_cell(dofhandler))
    f_int = zeros(total_dofs)
    f_int_sq = zeros(total_dofs)
    # Internal force vector 
    C_int = zeros(dim*dim)
    assembler = start_assemble(K, f_int)

    @timeit "assemble loop" begin
        @showprogress 0.2 "Assembling..." for element_id in 1:getncells(mesh)
            celldofs!(global_dofs, dofhandler, element_id)

            element_coords = getcoordinates(mesh, element_id)

            u_e = u[global_dofs]
            un_e = un[global_dofs]

            ele_matstats = view(mss, :, element_id)
            temp_matstats = view(temp_mss, :, element_id)

            fe(field) = intf(problem, field, un_e, element_coords,
                            fev_u, fev_ξ, dt, ele_matstats, temp_matstats, mps[polys[element_id]], true)
            @timeit "intf" begin
                fe_int, Ke = fe(u_e)
            end

            @timeit "assemble to global" begin
                JuAFEM.assemble!(assembler, fe_int, Ke, global_dofs)
                JuAFEM.assemble!(f_int_sq, fe_int.^2, global_dofs)
            end
        end
    end

    return K, f_int, f_int_sq
end
