import Optim

fff = open("bcs", "w")
function solve_problem{dim}(problem::AbstractProblem, mesh::GeometryMesh, dofhandler::DofHandler, dbcs::DirichletBoundaryConditions,
                            fe_values::FEValues{dim}, mps::Vector, timesteps, exporter, polys)

    K = create_sparsity_pattern(dofhandler)
    free = free_dofs(dbcs)
    ∆u = 0.0000001 * ones(length(free))
    primary_field = zeros(ndofs(dofhandler))
    prev_primary_field = copy(primary_field)
    full_residuals = zeros(ndofs(dofhandler))
    test_field = copy(primary_field)
    ddx_full = zeros(ndofs(dofhandler))
    n_qpoints = length(points(get_quadrule(fe_values)))


    nslip = length(mps[1].angles)
    if isa(problem, PrimalProblem)
        mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:nelements(mesh)]
        temp_mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:nelements(mesh)]
    else
        mss = [CrystPlastDualQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:nelements(mesh)]
        temp_mss = [CrystPlastDualQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:nelements(mesh)]
    end

    n_basefuncs = n_basefunctions(get_functionspace(fe_values))


    t_prev = timesteps[1]
    for (nstep, t) in enumerate(timesteps[2:end])
        dt = t - t_prev
        t_prev = t

        println("Step $nstep, t = $t, dt = $dt")

        update_dirichletbcs!(dbcs, t)
        println(fff, dbcs.values)
        # Make primary field obey BC
        apply!(primary_field, dbcs)
        copy!(test_field, primary_field)


        iter = 1
        n_iters = 20

        #=
        function f!(∆u, fvec)
            test_field[free] = primary_field[free] + ∆u
            K_condensed, f = assemble!(problem, K, test_field, prev_primary_field, fe_values,
                                       mesh, dofhandler, dbcs, mps, mss, temp_mss, dt, polys)
            print("Error:")
            full_residuals[free] = f
             print_residuals(dofhandler, full_residuals)

           # print("Total:")
          #  print_residuals(dofhandler, test_field)

            #println("|f|: ", norm(f), " |ddx|: ", norm(ddx))
            exporter(t, test_field, mss)

            copy!(fvec, f)
        end

        function g!(∆u, K2)
            test_field[free] = primary_field[free] + ∆u
            K_condensed, f = assemble!(problem, K, test_field, prev_primary_field, fe_values,
                                       mesh, dofhandler, dbcs, mps, mss, temp_mss, dt, polys)
            copy!(K2, K_condensed)
        end

        df = DifferentiableSparseMultivariateFunction(f!, g!)

        nlsolve(df, ∆u, method=:trust_region)#, linesearch! = Optim.backtracking_linesearch!)
        =#


        while iter == 1 || norm(f, Inf)  >= 1e-4
            test_field[free] = primary_field[free] + ∆u
            K_condensed, f = assemble!(problem, K, test_field, prev_primary_field, fe_values,
                                       mesh, dofhandler, dbcs, mps, mss, temp_mss, dt, polys)
            ddx = K_condensed \ f
            ∆u -= ddx

            full_residuals[free] = f
            print("Error:")
            print_residuals(dofhandler, full_residuals)

            print("Increment")
            ddx_full[free] = ddx
            print_residuals(dofhandler, ddx_full)
            #print("Total:")
            #print_residuals(dofhandler, test_field)

            #println("|f|: ", norm(f), " |ddx|: ", norm(ddx))
            iter += 1
            #exporter(t, test_field, mss)

            #@timer "factorization" ∆u -=  cholfact(Symmetric(K_condensed, :U)) \ f
        end



        #
        copy!(primary_field, test_field)
        copy!(prev_primary_field, primary_field)
        mss, temp_mss = temp_mss, mss
        exporter(t, primary_field, mss)
    end
    return primary_field, mss
end

function assemble!{dim}(problem, K::SparseMatrixCSC, primary_field::Vector, prev_primary_field::Vector,
                        fe_values::FEValues{dim}, mesh::GeometryMesh, dofhandler::DofHandler,
                        bcs::DirichletBoundaryConditions, mps::Vector, mss, temp_mss, dt::Float64, polys::Vector{Int})

    dofs_per_element = 20
    chunk_size = 20
    #@assert length(primary_field) == length(dofs.dof_ids)
    fill!(K.nzval, 0.0)

    dofs_per_element = length(dofs_element(dofhandler, 1))
    f_int = zeros(ndofs(dofhandler))
    Ke = zeros(dofs_per_element, dofs_per_element)

    #e_coordinates = zeros(dim, n_basefuncs)

  #chunk_size = 20, needs to parameterize assemble according to chunk_size?
 #G = ForwardDiff.workvec_eltype(ForwardDiff.GradientNumber, Float64, Val{dofs_per_element}, Val{dofs_per_element})
 #nslip = length(mps[1].angles)
 #n_basefuncs = n_basefunctions(get_functionspace(fe_values))
 # #TODO: Refactor this into a type
 #fe_u = Vec{dim, G}[zero(Vec{dim, G}) for i in 1:n_basefuncs]
 #fe_g = [zeros(G, n_basefuncs) for i in 1:nslip]
 #fe_go = [zeros(G, n_basefuncs) for i in 1:nslip]
 #fe_uF64 = [zero(Vec{dim, Float64}) for i in 1:n_basefuncs]
 #fe_gF64 = [zeros(Float64, n_basefuncs) for i in 1:nslip]
 #fe_goF64 = [zeros(Float64, n_basefuncs) for i in 1:nslip]

 #local prev_primary_element_field
 #local ele_matstats
 #local temp_matstats
 #local element_coords
 #local mp

 # fe(field) = intf(field, prev_primary_element_field, element_coords, fe_values, fe_u, fe_g, fe_go, dt, ele_matstats, temp_matstats, mp)

 # Ke! = ForwardDiff.jacobian(fe, mutates = true, chunk_size = dofs_per_element)

    for element_id in 1:nelements(mesh)
        edof = dofs_element(dofhandler, element_id)
        element_coords = element_coordinates(mesh, element_id)

        primary_element_field = primary_field[edof]
        prev_primary_element_field = prev_primary_field[edof]

        ele_matstats = view(mss, :, element_id)
        temp_matstats = view(temp_mss, :, element_id)

        fe(field) = intf(problem, field, prev_primary_element_field, element_coords,
                         fe_values, dt, ele_matstats, temp_matstats, mps[polys[element_id]], true)
        fe_int, Ke = fe(primary_element_field)


        #Ke = Calculus.jacobian(x -> fe(x)[1], primary_element_field, :central)

        #fe_int, Ke = fe(primary_element_field)

     #   fe_int = intf(primary_element_field, prev_primary_element_field, element_coords,
     #             fe_values, fe_uF64, fe_gF64, fe_goF64, dt, ele_matstats, temp_matstats, mps[polys[element_id]])
#
     #   mp = mps[polys[element_id]]
     #   _, allresults = Ke!(Ke, primary_element_field)


        #println(element_coords)
        #println(primary_element_field)
        #println(mps[polys[element_id]])
        #println(fe_int)
        #error()

        assemble!(f_int, fe_int, edof)
        assemble!(K, Ke, edof)

    end

    free = free_dofs(bcs)
    return K[free, free], f_int[free]
end
