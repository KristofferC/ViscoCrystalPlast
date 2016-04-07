include("quadrature_data.jl")
include("intf_primal.jl")


# f(x) -> v
function update_bcs!(mesh::GeometryMesh, dofs::Dofs, bc::DirichletBoundaryConditions, time::Float64, f)
    dofs_per_node = length(dofs.dof_types)
    for i in eachindex(bc.dof_ids)
        dof_type = bc.dof_types[i]
        dof_id = bc.dof_ids[i]
        node = div(dof_id + dofs_per_node -1, dofs_per_node)
        ViscoCrystalPlast.set_value(bc, f(dof_type, mesh.coords[:, node], time), i)
    end
end

function assemble!{dim}(K::SparseMatrixCSC, primary_field::Vector, prev_primary_field::Vector,
                        fe_values::FEValues{dim}, mesh::GeometryMesh, dofs::Dofs,
                        bcs::DirichletBoundaryConditions, mp, mss, dt)

    @assert length(primary_field) == length(dofs.dof_ids)
    fill!(K.nzval, 0.0)
    n_basefuncs = n_basefunctions(get_functionspace(fe_values))
    dofs_per_node = length(dofs.dof_types)
    dofs_per_element = n_basefuncs * dofs_per_node

    f_int = zeros(length(dofs.dof_ids))
    K_element = zeros(dofs_per_element, dofs_per_element)
    e_coordinates = zeros(dim, n_basefuncs)

    G = ForwardDiff.workvec_eltype(ForwardDiff.GradientNumber, Float64, Val{12}, Val{12})
    nslip = length(mp.angles)
    fe_u = [zero(Vec{dim, G}) for i in 1:n_basefuncs]
    fe_g = [zeros(G, n_basefuncs) for i in 1:nslip]


    local prev_primary_element_field
    local ele_matstats
    fe(field) = intf_opt(field, prev_primary_element_field, e_coordinates, fe_values, fe_u, fe_g, dt, ele_matstats, mp)


    Ke! = ForwardDiff.jacobian(fe, ForwardDiff.AllResults, mutates = true, chunk_size = 12)

    for element_id in 1:size(mesh.topology, 2)
        edof = dofs_element(mesh, dofs, element_id)
        element_coordinates!(e_coordinates , mesh, element_id)
        primary_element_field = primary_field[edof]
        prev_primary_element_field = prev_primary_field[edof]
        ele_matstats = slice(mss, :, element_id)

        _, allresults = Ke!(K_element, primary_element_field)
        fe_int = ForwardDiff.value(allresults)
        #print(K_element)
        #error("fhdsfd")


        @assert any(isnan, fe_int) == false  "nan in force $element_id, $edof, $ele_matstats"
        @assert any(isnan, K_element) == false "nan in stiffness $element_id $K_element, $primary_element_field, $e_coordinates"


        assemble!(f_int, ForwardDiff.value(allresults), edof)
        assemble!(K, K_element, edof)
    end

    free = setdiff(dofs.dof_ids, bcs.dof_ids)
    return K[free, free], f_int[free]
end


function solve_primal_problem{dim}(mesh, dofs, bcs, fe_values::FEValues{dim}, mp, boundary_f)
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
    mss = [CrystPlastPrimalQD(length(mp.angles), Dim{dim}) for i = 1:n_qpoints, j = 1:size(mesh.topology, 2)]

        nstep = 20
    t_end = 100
    dt = t_end / nstep # Constant time step


    step = 0
    t_prev = 0.0
    for i in 1:nstep
        t = i / nstep
        step += 1

        println("Step $step, t = $t, dt = $dt")

        update_bcs!(mesh, dofs, bcs, t, boundary_f)
        primary_field[bcs.dof_ids] = bcs.values
        copy!(test_field, primary_field)

        iter = 1
        n_iters = 20
        while iter == 1 || norm(f, Inf)  >= 1e-6
            test_field[free] = primary_field[free] + ∆u
            K_condensed, f = assemble!(K, test_field, prev_primary_field, fe_values, mesh, dofs, bcs, mp, mss, dt)
            #∆u -=  K_condensed \ f
            ∆u -=  cholfact(Symmetric(K_condensed, :U)) \ f
            println(norm(f))

            iter +=1
            if n_iters == iter
                error("too many iterations without convergence")
            end
        end

        copy!(prev_primary_field, primary_field)
        copy!(primary_field, test_field)
        t_prev = t

        #@timeit to "vtk" begin
        #    vtkoutput(pvd, t, step+=1, mesh, primary_field, dim)
       #end
    end
    #vtk_save(pvd)
    #print(to)
end




#=
function vtkoutput{dim}(pvd, time, tstep, mesh, u, mss, quad_rule::QuadratureRule{dim}, nslips)
    @unpack mesh: coord, dof, edof, ex, ey, topology
    nnodes = dim == 2 ? 3 : 4
    nrelem = nelems(mesh)
    udofs = u_dofs(dim, nnodes, 1, nslips)
    vtkfile = vtk_grid(topology, coord, "vtks/box_primal_$tstep")

    τs_di = zeros(nslips, nrelem)
    τs = zeros(nslips, nrelem)

    tot_weights = sum(quad_rule.weights)

    εs = [zero(SymmetricTensor{2,dim}) for i in 1:nrelem]
    σs = [zero(SymmetricTensor{2,dim}) for i in 1:nrelem]
    ε_ps = [zero(SymmetricTensor{2,dim}) for i in 1:nrelem]
    gs = [[zero(Vec{dim,Float64}) for i in 1:nrelem] for α in 1:nslips]
    for i in 1:nrelem
        for q_point in 1:length(JuAFEM.points(quad_rule))
            w = quad_rule.weights[q_point] / tot_weights
            σs[i] += mss[q_point, i].σ * w
            εs[i] += mss[q_point, i].ε * w
            ε_ps[i] += mss[q_point, i].ε_p * w
            for α = 1:nslips
                τs_di[α, i] += mss[q_point, i].τ_di[α] * w
                τs[α, i] += mss[q_point, i].τ[α] * w
                gs[α][i] += mss[q_point, i].g[α] * w
            end
        end
    end

    for α in 1:nslips
        vtk_cell_data(vtkfile, reinterpret(Float64, gs[α], (dim, length(gs[α]))), "g_$α")
        vtk_cell_data(vtkfile, vec(τs[α, :]), "Schmid stress_$α")
        vtk_cell_data(vtkfile, vec(τs_di[α, :]), "Tau dissip_$α")
    end

    dofs_u = mesh.dof[1:dim, :]

    disp = u[dofs_u]
    disp = reshape(disp, size(coord))
    if dim == 2
        disp = [disp; zeros(nnodes)']
    end

    for i in 1:nslips
        dofs_g = mesh.dof[dim+i, :]
        grad = u[dofs_g]
        vtk_point_data(vtkfile, grad, "slip_$i")
    end

    vtk_point_data(vtkfile, disp, "displacement")
    vtk_cell_data(vtkfile, reinterpret(Float64, εs, (6, length(εs))) ,"Strain")
    vtk_cell_data(vtkfile, reinterpret(Float64, σs, (6, length(σs))), "Stress")
    vtk_cell_data(vtkfile, reinterpret(Float64, ε_ps, (6, length(ε_ps))), "Plastic strain")
    collection_add_timestep(pvd, vtkfile, time)
end

function total_slip{T}(mesh, u::Vector{T}, mss, quad_rule, function_space, nslip, mp)
    fev = FEValues(Float64, quad_rule, function_space)
    γs = Vector{Vector{T}}(nslip)
    tot_slip = 0.0
    tot_grad_en = 0.0
    ngradvars = 1
    n_basefuncs = n_basefunctions(get_functionspace(fev))
    nnodes = n_basefuncs
    ed = extract(mesh.edof, u)

    Hg = [mp.H⟂ * mp.s[α] ⊗ mp.s[α] for α in 1:nslip]

    for i in 1:nelems(mesh)
        ug = ed[:, i]
        x = [mesh.ex[:, i] mesh.ey[:, i]]'
        x_vec = reinterpret(Vec{2, T}, x, (nnodes,))
        reinit!(fev, x_vec)
        for α in 1:nslip
            gd = g_dofs(ndim, nnodes, ngradvars, nslip, α)
            γs[α] = ug[gd]
        end

        for q_point in 1:length(quad_rule.points)
            for α = 1:nslip
                g = mss[q_point, i].g[α]
                tot_grad_en += 0.5 * mp.l^2 * g ⋅ (Hg[α] ⋅ g) * detJdV(fev, q_point)
                γ = function_scalar_value(fev, q_point, γs[α])
                tot_slip += abs2(γ * detJdV(fev, q_point))
            end
        end
    end
    println("total nodes: $(size(mesh.edof, 2))")
    println("total_slip = $(sqrt(tot_slip))")
    println("total_grad = $(tot_grad_en)")
end
=#