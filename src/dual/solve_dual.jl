include("quadrature_data.jl")
include("intf_dual.jl")



#=
function solve_dual_problem{dim}(mesh, dofs, bcs, fe_values::FEValues{dim}, mp, timesteps, boundary_f)
    free = setdiff(dofs.dof_ids, bcs.dof_ids)

    K = create_sparsity_pattern(mesh, dofs)
    ∆u = 0.000 * ones(length(free))
    primary_field = zeros(length(dofs.dof_ids))
    test_field = copy(primary_field)
    pvd = paraview_collection("vtks/shear_dual")

    n_basefuncs = n_basefunctions(get_functionspace(fe_values))
    nslip = length(mp.angles)

    # Initialize Quadrature Data
    n_qpoints = length(points(get_quadrule(fe_values)))
    mss = [CrystPlastDualQD(length(mp.angles), Dim{dim}) for i = 1:n_qpoints, j = 1:size(mesh.topology, 2)]
    temp_mss = [CrystPlastDualQD(length(mp.angles), Dim{dim}) for i = 1:n_qpoints, j = 1:size(mesh.topology, 2)]

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
            @timer "total K" K_condensed, f = assemble!(K, test_field, fe_values, mesh, dofs, bcs, mp, mss, temp_mss, dt)
            @timer "factorization"  ∆u -=  K_condensed \ f
            #@timer "factorization" ∆u -=  cholfact(Symmetric(K_condensed, :U)) \ f
            println(norm(f))

            iter +=1
            if n_iters == iter
                error("too many iterations without convergence")
            end
        end

        copy!(primary_field, test_field)
        mss, temp_mss = temp_mss, mss

        #@timeit to "vtk" begin
        #    vtkoutput(pvd, t, step+=1, mesh, primary_field, dim)
       #end
    end
    #vtk_save(pvd)
    #print(to)
end
=#