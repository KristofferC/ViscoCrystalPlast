
function solve_problem{dim}(problem::AbstractProblem, mesh, dofs, bcs, fe_values::FEValues{dim}, mp, timesteps, boundary_f)
    free = setdiff(dofs.dof_ids, bcs.dof_ids)

    K = create_sparsity_pattern(mesh, dofs)
    ∆u = 0.0001 * ones(length(free))
    primary_field = zeros(length(dofs.dof_ids))
    prev_primary_field = copy(primary_field)
    test_field = copy(primary_field)
    pvd = paraview_collection("vtks/shear_dual")

    n_basefuncs = n_basefunctions(get_functionspace(fe_values))
    nslip = length(mp.angles)

    # Initialize Quadrature Data
    n_qpoints = length(points(get_quadrule(fe_values)))
    if isa(problem, PrimalProblem)
        mss = [CrystPlastPrimalQD(length(mp.angles), Dim{dim}) for i = 1:n_qpoints, j = 1:size(mesh.topology, 2)]
        temp_mss = [CrystPlastPrimalQD(length(mp.angles), Dim{dim}) for i = 1:n_qpoints, j = 1:size(mesh.topology, 2)]
    else
        mss = [CrystPlastDualQD(length(mp.angles), Dim{dim}) for i = 1:n_qpoints, j = 1:size(mesh.topology, 2)]
        temp_mss = [CrystPlastDualQD(length(mp.angles), Dim{dim}) for i = 1:n_qpoints, j = 1:size(mesh.topology, 2)]
    end


    t_prev = timesteps[1]
    for (nstep, t) in enumerate(timesteps[2:end])
        dt = t - t_prev
        t_prev = t

        println("Step $nstep, t = $t, dt = $dt")

        update_bcs!(mesh, dofs, bcs, t, boundary_f)
        primary_field[bcs.dof_ids] = bcs.values
        copy!(test_field, primary_field)

        iter = 1
        n_iters = 20
        while iter == 1 || norm(f, Inf)  >= 1e-6
            test_field[free] = primary_field[free] + ∆u

                @timer "total K" K_condensed, f = assemble!(K, test_field, prev_primary_field, fe_values,
                                                            mesh, dofs, bcs, mp, mss, temp_mss, dt)

            @timer "factorization"  ∆u -=  K_condensed \ f
            #@timer "factorization" ∆u -=  cholfact(Symmetric(K_condensed, :U)) \ f

            println(norm(f))

            iter +=1
            if n_iters == iter
                error("too many iterations without convergence")
            end
        end

        copy!(prev_primary_field, primary_field)
        copy!(primary_field, test_field)
        mss, temp_mss = temp_mss, mss

        #@timeit to "vtk" begin
        #    vtkoutput(pvd, t, step+=1, mesh, primary_field, dim)
       #end
    end
    #vtk_save(pvd)
    #print(to)
end

function assemble!{dim}(K::SparseMatrixCSC, primary_field::Vector, prev_primary_field::Vector,
                        fe_values::FEValues{dim}, mesh::GeometryMesh, dofs::Dofs,
                        bcs::DirichletBoundaryConditions, mp, mss, temp_mss, dt)

    @assert length(primary_field) == length(dofs.dof_ids)
    fill!(K.nzval, 0.0)

    n_basefuncs = n_basefunctions(get_functionspace(fe_values))
    dofs_per_node = length(dofs.dof_types)
    dofs_per_element = n_basefuncs * dofs_per_node

    f_int = zeros(length(dofs.dof_ids))
    K_element = zeros(dofs_per_element, dofs_per_element)
    e_coordinates = zeros(dim, n_basefuncs)

   # chunk_size = 20
    G = ForwardDiff.workvec_eltype(ForwardDiff.GradientNumber, Float64, Val{12}, Val{12})
    nslip = length(mp.angles)
    fe_u = [zero(Vec{dim, G}) for i in 1:n_basefuncs]
    fe_g = [zeros(G, n_basefuncs) for i in 1:nslip]


    fe_uF64 = [zero(Vec{dim, Float64}) for i in 1:n_basefuncs]
    fe_gF64 = [zeros(Float64, n_basefuncs) for i in 1:nslip]


    local prev_primary_element_field
    local ele_matstats
    local temp_matstats
    fe(field) = intf(field, prev_primary_element_field, e_coordinates, fe_values, fe_u, fe_g, dt, ele_matstats, temp_matstats, mp)

    Ke! = ForwardDiff.jacobian(fe, ForwardDiff.AllResults, mutates = true, chunk_size = 12)

    for element_id in 1:size(mesh.topology, 2)
        edof = dofs_element(mesh, dofs, element_id)
        element_coordinates!(e_coordinates , mesh, element_id)
        primary_element_field = primary_field[edof]
        prev_primary_element_field = prev_primary_field[edof]
        ele_matstats = slice(mss, :, element_id)
        temp_matstats = slice(temp_mss, :, element_id)
        intf(primary_element_field, prev_primary_element_field, e_coordinates,
                  fe_values, fe_uF64, fe_gF64, dt, ele_matstats, temp_matstats, mp)

        @timer "call Ke" _, allresults = Ke!(K_element, primary_element_field)
        fe_int = ForwardDiff.value(allresults)

        @assert any(isnan, fe_int) == false  "nan in force $element_id, $edof, $ele_matstats"
        @assert any(isnan, K_element) == false "nan in stiffness $element_id $K_element, $primary_element_field, $e_coordinates"

        assemble!(f_int, ForwardDiff.value(allresults), edof)
        @timer "assemble_stiffness" assemble!(K, K_element, edof)
    end

    free = setdiff(dofs.dof_ids, bcs.dof_ids)
    return K[free, free], f_int[free]
end