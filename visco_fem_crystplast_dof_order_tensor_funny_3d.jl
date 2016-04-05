using MAT
using JuAFEM
using Parameters
using WriteVTK
using NLsolve
using Devectorize
using CALFEM

include("common.jl")

include("visco_crystal_plast_dof_order_tensor.jl")

include("../meshes/read_mph.jl")

function update_bcs!(bc, B_u, B_g, mesh, nslip, i, nstep, dim)
    @unpack mesh: coord, dof
    cx = 0.02 * i / nstep
    cy = 0.01 * i / nstep

    dofs_per_node = dim + nslip
    for k in 1:size(B_u, 1)
        dof = B_u[k]
        node = div(dof+dofs_per_node -1, dofs_per_node)
        bc[k, 1] = dof

        bc[k, 2] = cy * coord[2, node]
    end

    for (i, k) in enumerate(length(B_u)+1:length(B_u) + length(B_g))
        dof = B_g[i]
        bc[k, 1] = dof
        bc[k, 2] = 0.0
    end
end

function analyze()
    mp = setup_material(Dim{3})
    nslip = length(mp.angles)

    mesh, B_u, B_g = setup_geometry("../meshes/3d_cube.mphtxt", nslip, 1, Dim{3})
    @unpack mesh: coord, dof, edof, ex, ey
    bc = zeros(length(B_u) + length(B_g), 2)


    pvd = paraview_collection("vtks/shear_primal")

    # Analysis parameters
    nstep = 3
    t_end = 100
    dt = t_end / nstep # Constant time step

    # Finite Element
    function_space = JuAFEM.Lagrange{3, RefTetrahedron, 1}()
    q_rule = QuadratureRule(Dim{3}, RefTetrahedron(), 1)

    # Initialize Material States
    mss = [CrystPlastMS(nslip, 3) for i = 1:length(JuAFEM.points(q_rule)), j = 1:nelems(mesh)]

    u = zeros(ndofs(mesh))
    u_prev = zeros(ndofs(mesh))
    for i in 1:nstep

        update_bcs!(bc, B_u, B_g, mesh, nslip, i, nstep, 3)
        free, fixed = get_freefixed(mesh, bc)

        # Set dirichlet boundary conditions on dofs
        u[fixed] = bc[:, 2]

        # Guess zero (for now)
        ∆u0 = rand(length(free))

        function ∆f!(du, fvec)
            fint = tstep(du, u, u_prev, mesh, dt, mp, function_space, q_rule, mss, free)
            copy!(fvec, fint)
        end

        function g!(du, gjac)
            K = tstep_grad(du, u, u_prev, mesh, dt, mp, function_space, q_rule, mss, free)
            # Update
            gjac.colptr = K.colptr
            gjac.rowval = K.rowval
            gjac.nzval = K.nzval
        end

        gjac = spzeros(length(free), length(free))
        fvec = ones(length(free))
        n_iters = 15
        iter = 1
        while norm(fvec, Inf) >= 1e-6
            ∆f!(∆u0 , fvec)
            g!(∆u0 , gjac)
            ∆u0 -=  gjac \ fvec
            println(norm(fvec, Inf))
            iter +=1
            if n_iters == iter
                error("fuck!")
            end
        end

        @devec u[free] += ∆u0

        copy!(u_prev, u)

        vtkoutput(pvd, t_end / nstep * i, i, mesh, u, mss, q_rule, nslip)
    end
    #total_slip(mesh, u, mss, q_rule, function_space, nslip, mp)
    @label done
    vtk_save(pvd)
end


function tstep(∆u_red, u, u_prev, mesh, dt, mp, function_space, q_rule, mss, free)
    @unpack mesh: dof, edof, ex, ey, ez

    ∆u = zeros(ndofs(mesh))
    ∆u[free] = ∆u_red
    ed = extract(edof, u + ∆u)
    ed_prev = extract(edof, u_prev)
    fint = zeros(ndofs(mesh))

    fev = FEValues(Float64, q_rule, function_space)

    for i = 1:nelems(mesh) # Element contributions
        ug = ed[:, i]
        ug_prev = ed_prev[:, i]
        x = [ex[:, i] ey[:, i] ez[:, i]]'
        fe = intf_opt(ug, ug_prev, x, fev, dt, slice(mss, :, i), mp)
        fint[edof[:, i]] += fe
    end

    return fint[free]
end


function tstep_grad(∆u_red, u, u_prev, mesh, dt, mp, function_space, q_rule, mss_grad, free)
    @unpack mesh: dof, edof, ex, ey

    ∆u = zeros(ndofs(mesh))
    ∆u[free] = ∆u_red
    ed = extract(edof, u + ∆u)
    ed_prev = extract(edof, u_prev)


    fev = FEValues(Float64, q_rule, function_space)

    a = start_assemble()

    for i = 1:nelems(mesh)
        ug = ed[:, i]
        ug_prev = ed_prev[:, i]
        x = [mesh.ex[:, i] mesh.ey[:, i] mesh.ez[:, i]]'
        f(uu) = intf_opt(uu, ug_prev, x, fev, dt, slice(mss_grad, :, i), mp)
        Ke = ForwardDiff.jacobian(f, ug, chunk_size=10)
        assemble(edof[:, i], a, Ke)
    end
    K = end_assemble(a)

    return K[free, free]
end

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