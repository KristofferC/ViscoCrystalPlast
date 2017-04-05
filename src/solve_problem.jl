function solve_problem{dim}(problem::AbstractProblem, mesh::Grid, dofhandler::DofHandler, dbcs::DirichletBoundaryConditions,
                            fev_u::CellVectorValues{dim}, fev_ξ::CellScalarValues{dim}, mps::Vector, timesteps, exporter, polys)

    println("Analysis started\n----------------")
    println("Time steps: ", length(timesteps), " steps [", first(timesteps), ",", last(timesteps), "]")
    println("Mesh: n_elements: ", getncells(mesh), ", n_nodes:", getnnodes(mesh))
    println("Dofs: n_dofs: ", ndofs(dofhandler))

    @timeit "sparsity pattern" begin
        K = create_sparsity_pattern(dofhandler)
    end

    free = dbcs.free_dofs
    ∆u = 0.0001 * rand(length(free))
    primary_field = zeros(ndofs(dofhandler))
    prev_primary_field = copy(primary_field)
    full_residuals = zeros(ndofs(dofhandler))
    test_field = copy(primary_field)
    ddx_full = zeros(ndofs(dofhandler))
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

        # Make primary field obey BC
        apply!(primary_field, dbcs)
        copy!(test_field, primary_field)

        iter = 1
        n_iters = 40

        while iter == 1 || norm(f./f_sq, Inf)  >= 1e-5 && norm(f, Inf) >= 1e-8
            test_field[free] = primary_field[free] + ∆u
            @timeit "assemble" begin
                K_condensed, f, f_sq = assemble!(problem, K, test_field, prev_primary_field, fev_u, fev_ξ,
                                                 mesh, dofhandler, dbcs, mps, mss, temp_mss, dt, polys)
            end
            @timeit "factorization" begin
                ddx = solveMUMPS(K_condensed,f,1,1);
            end
            ∆u -= ddx

            full_residuals[free] = f
            #println("Error: ", norm(f, Inf))
            println("Step: $nstep, iter: $iter")
            print("Error: ")
            print_residuals(dofhandler, full_residuals)

            iter += 1
        end

        copy!(primary_field, test_field)
        copy!(prev_primary_field, primary_field)

        # temp_mss is now the true matstats
        mss, temp_mss = temp_mss, mss
        exporter(t, primary_field, mss)
    end
    return primary_field, mss
end

function assemble!{dim}(problem, K::SparseMatrixCSC, primary_field::Vector, prev_primary_field::Vector,
                        fev_u::CellVectorValues{dim}, fev_ξ::CellScalarValues{dim}, mesh::Grid, dofhandler::DofHandler,
                        bcs::DirichletBoundaryConditions, mps::Vector, mss, temp_mss, dt::Float64, polys::Vector{Int})

    #@assert length(primary_field) == length(dofs.dof_ids)

    fill!(K.nzval, 0.0)
    global_dofs = zeros(Int, ndofs_per_cell(dofhandler))
    f_int = zeros(ndofs(dofhandler))
    f_int_sq = zeros(ndofs(dofhandler))
    assembler = start_assemble(K, f_int)

    @timeit "assemble loop" begin
        @showprogress 0.2 "Assembling..." for element_id in 1:getncells(mesh)
            celldofs!(global_dofs, dofhandler, element_id)

            element_coords = getcoordinates(mesh, element_id)

            primary_element_field = primary_field[global_dofs]
            prev_primary_element_field = prev_primary_field[global_dofs]

            ele_matstats = view(mss, :, element_id)
            temp_matstats = view(temp_mss, :, element_id)

            fe(field) = intf(problem, field, prev_primary_element_field, element_coords,
                            fev_u, fev_ξ, dt, ele_matstats, temp_matstats, mps[polys[element_id]], true)
            @timeit "intf" begin
                fe_int, Ke = fe(primary_element_field)
            end

            @timeit "assemble to global" begin
                JuAFEM.assemble!(assembler, fe_int, Ke, global_dofs)
                JuAFEM.assemble!(f_int_sq, fe_int.^2, global_dofs)
            end
        end
    end

    @timeit "extract free" begin
        free = bcs.free_dofs
        Kfree = K[free, free]
        f_int_free = f_int[free]
        f_sq_free = sqrt.(f_int_sq[free])
    end

    return Kfree, f_int_free, f_sq_free
end
