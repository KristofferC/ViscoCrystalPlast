function solve_problem{dim}(problem::AbstractProblem, mesh, dofs, bcs, fe_values::FEValues{dim}, mp, timesteps, boundary_f, exporter)
    free = setdiff(dofs.dof_ids, bcs.dof_ids)

    K = create_sparsity_pattern(mesh, dofs)
    ∆u = 0.0001 * ones(length(free))
    primary_field = zeros(length(dofs.dof_ids))
    prev_primary_field = copy(primary_field)
    test_field = copy(primary_field)

    n_qpoints = length(points(get_quadrule(fe_values)))
    n_elements = size(mesh.topology, 2)
    nslip = length(mp.angles)
    if isa(problem, PrimalProblem)
        mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:n_elements]
        temp_mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:n_elements]
    else
        mss = [CrystPlastDualQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:n_elements]
        temp_mss = [CrystPlastDualQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:n_elements]
    end

    n_basefuncs = n_basefunctions(get_functionspace(fe_values))
    dofs_per_node = length(dofs.dof_types)
    dofs_per_element = n_basefuncs * dofs_per_node

    t_prev = timesteps[1]
    for (nstep, t) in enumerate(timesteps[2:end])
        dt = t - t_prev
        t_prev = t

        println("Step $nstep, t = $t, dt = $dt")

        update_bcs!(mesh, dofs, bcs, t, boundary_f)

        # Make primary field obey BC
        apply!(primary_field, bcs)
        copy!(test_field, primary_field)

        iter = 1
        n_iters = 20

        while iter == 1 || norm(f, Inf)  >= 1e-6

            test_field[free] = primary_field[free] + ∆u
            K_condensed, f = assemble!(problem, K, test_field, prev_primary_field, fe_values,
                                       mesh, dofs, bcs, mp, mss, temp_mss, dt, Val{dofs_per_element})
            ddx = K_condensed \ f
            ∆u -= ddx
            println("|f|: ", norm(f), " |ddx|: ", norm(ddx))
            iter += 1
            #@timer "factorization" ∆u -=  cholfact(Symmetric(K_condensed, :U)) \ f
        end

        #
        copy!(primary_field, test_field)
        copy!(prev_primary_field, primary_field)
        copy!(primary_field, test_field)
        mss, temp_mss = temp_mss, mss
        exporter(t, primary_field, mss)
    end
    return primary_field, mss
end

function assemble!{dim, dofs_per_element}(problem, K::SparseMatrixCSC, primary_field::Vector, prev_primary_field::Vector,
                        fe_values::FEValues{dim}, mesh::GeometryMesh, dofs::Dofs,
                        bcs::DirichletBoundaryConditions, mp, mss, temp_mss, dt, ::Type{Val{dofs_per_element}})

    @assert length(primary_field) == length(dofs.dof_ids)
    fill!(K.nzval, 0.0)

    n_basefuncs = n_basefunctions(get_functionspace(fe_values))
    dofs_per_node = length(dofs.dof_types)


    f_int = zeros(length(dofs.dof_ids))
    K_element = zeros(dofs_per_element, dofs_per_element)
    e_coordinates = zeros(dim, n_basefuncs)

   # chunk_size = 20, needs to parameterize assemble according to chunk_size?
    G = ForwardDiff.workvec_eltype(ForwardDiff.GradientNumber, Float64, Val{dofs_per_element}, Val{dofs_per_element})
    nslip = length(mp.angles)

    # TODO: Refactor this into a type
    fe_u = [zero(Vec{dim, G}) for i in 1:n_basefuncs]
    fe_g = [zeros(G, n_basefuncs) for i in 1:nslip]
    fe_go = [zeros(G, n_basefuncs) for i in 1:nslip]
    fe_uF64 = [zero(Vec{dim, Float64}) for i in 1:n_basefuncs]
    fe_gF64 = [zeros(Float64, n_basefuncs) for i in 1:nslip]
    fe_goF64 = [zeros(Float64, n_basefuncs) for i in 1:nslip]

   local prev_primary_element_field
   local ele_matstats
   local temp_matstats

    fe(field) = intf_dual(field, prev_primary_element_field, e_coordinates, fe_values, fe_u, fe_g, fe_go, dt, ele_matstats, temp_matstats, mp)
    Ke! = ForwardDiff.jacobian(fe, mutates = true, chunk_size = dofs_per_element)

    for element_id in 1:size(mesh.topology, 2)
        edof = dofs_element(mesh, dofs, element_id)

        element_coordinates!(e_coordinates , mesh, element_id)
        primary_element_field = primary_field[edof]
        prev_primary_element_field = prev_primary_field[edof]

        ele_matstats = slice(mss, :, element_id)
        temp_matstats = slice(temp_mss, :, element_id)

        fe_int2 = intf_dual(primary_element_field, prev_primary_element_field, e_coordinates,
                  fe_values, fe_uF64, fe_gF64, fe_goF64, dt, ele_matstats, temp_matstats, mp)

       # fe_int, Ke = intf(problem, primary_element_field, prev_primary_element_field, e_coordinates,
        #          fe_values, dt, ele_matstats, temp_matstats, mp)

        _, allresults = Ke!(K_element, primary_element_field)


        assemble!(f_int, fe_int2, edof)
        assemble!(K, K_element, edof)

    end

     free = setdiff(dofs.dof_ids, bcs.dof_ids)
    return K[free, free], f_int[free]
end


