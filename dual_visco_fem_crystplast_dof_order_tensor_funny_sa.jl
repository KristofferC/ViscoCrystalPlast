using MAT
using JuAFEM
using Parameters
using WriteVTK
using Devectorize
using NLsolve

include("common.jl")

include("dual_visco_crystal_plast_tensor_sa.jl")
include("../meshes/read_mph.jl")



function update_bcs!(bc, bc_dofs, mesh, i, nstep)
    @unpack mesh: coord, dof
    cx = 0.02 * i / nstep
    cy = 0.01 * i / nstep

    for k in 1:size(bc, 1)
        dof = bc_dofs[k]
        node = div(dof+2 + nslip - 1, 2 + nslip)
        bc[k, 1] = dof

        bc[k, 2] = cy * coord[2, node]
    end
end

vm(x) = sqrt(3/2) * vnorm(dev(x))


function analyze()
    nslip = 2
    #or q in 1:5
        mesh, B_u, B_g = setup_geometry("../meshes/test_mesh_4.mphtxt", nslip, 1)

        @unpack mesh: coord, dof, edof, ex, ey

        bc = zeros(length(B_u), 2)
        mp = setup_material()
        pvd = paraview_collection("vtks/shear_dual")

        # Analysis parameters
        nstep = 3
        t_end = 100
        dt = t_end / nstep # Constant time step

        # Finite Element
        function_space = JuAFEM.Lagrange{2, JuAFEM.Triangle, 1}()
        q_rule = JuAFEM.get_gaussrule(Dim{2}, JuAFEM.Triangle(), 1)


        # Initialize Material States
        mss = [CrystPlastMS(nslip, 2) for i = 1:length(JuAFEM.points(q_rule)), j = 1:nelems(mesh)]
        temp_mss = [CrystPlastMS(nslip, 2) for i = 1:length(JuAFEM.points(q_rule)), j = 1:nelems(mesh)]

        u = zeros(ndofs(mesh))
        for i in 1:nstep

            update_bcs!(bc, B_u, mesh, i, nstep)
            free, fixed = get_freefixed(mesh, bc)

            # Set dirichlet boundary conditions on dofs
            u[fixed] = bc[:, 2]

            # Guess zero (for now)
            ∆u0 = zeros(length(free))

            function ∆f!(du, fvec)
                fint = tstep(du, u, mesh, dt, mp, function_space, q_rule, mss, temp_mss, free)
                copy!(fvec, fint)
            end

            function g!(du, gjac)
                K = tstep_grad(du, u, mesh, dt, mp, function_space, q_rule, mss, temp_mss, free)
                # Update
                gjac.colptr = K.colptr
                gjac.rowval = K.rowval
                gjac.nzval = K.nzval
            end

            gjac = spzeros(length(free), length(free))
            fvec = ones(length(free))
            n_iters = 25
            iter = 1
            while norm(fvec, Inf) >= 1e-5
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

            mss, temp_mss = temp_mss, mss

            vtkoutput(pvd, t_end / nstep * i, i, mesh, u, mss, q_rule)
        end
        total_slip(mesh, u, mss, q_rule, function_space, nslip, mp)
        @label endit
        vtk_save(pvd)
    #end
end


function tstep(∆u_red, u,  mesh, dt, mp, function_space, q_rule, mss, temp_mss, free)
    @unpack mesh: dof, edof, ex, ey

    ∆u = zeros(ndofs(mesh))
    ∆u[free] = ∆u_red
    ed = extract(edof, u + ∆u)

    fint = zeros(ndofs(mesh))

    fev = FEValues(Float64, q_rule, function_space)

    for i = 1:nelems(mesh) # Element contributions
        ug = ed[:, i]
        x = [mesh.ex[:, i] mesh.ey[:, i]]'
        fe = intf_opt(ug, x, fev, dt, slice(mss, :, i), slice(temp_mss, :, i), mp)
        fint[edof[:, i]] += fe
    end

    return fint[free]
end

const FE_CACHE = ForwardDiff.JacobianCache(Val{12}, Val{6}, 12, Float64)

function tstep_grad(∆u_red, u, mesh, dt, mp, function_space, q_rule, mss, temp_mss, free)
    @unpack mesh: dof, edof, ex, ey

    ∆u = zeros(ndofs(mesh))
    ∆u[free] = ∆u_red
    ed = extract(edof, u + ∆u)

    fev = FEValues(Float64, q_rule, function_space)

    a = start_assemble()

    for i = 1:nelems(mesh)
        ug = ed[:, i]
        x = [mesh.ex[:, i] mesh.ey[:, i]]'
        f(ug) = intf_opt(ug, x, fev, dt, slice(mss, :, i), slice(temp_mss, :, i), mp)
        Ke = ForwardDiff.jacobian(f, ug, chunk_size=6, cache=FE_CACHE)
        assemble(edof[:, i], a, Ke)
    end
    K = end_assemble(a)

    return K[free, free]
end

function vtkoutput(pvd, time, tstep, mesh, u, mss, quad_rule)
    @unpack mesh: coord, dof, edof, ex, ey
    nnodes = size(dof, 2)
    nrelem = nelems(mesh)

    nslips = 2

    udofs = u_dofs(2, 3, 1, 2)
    vtkfile = vtk_grid(edof[udofs,:], coord, dof[1:2,:], 3, "vtks/box_$tstep")

    dofs_u = mesh.dof[1:2, :]

    disp = u[dofs_u]
    disp = reshape(disp, (2, nnodes))
    disp = [disp; zeros(nnodes)']


    τs_di = zeros(nslips, nrelem)
    τs = zeros(nslips, nrelem)
    γs = zeros(nslips, nrelem)
    χs = zeros(nslips, nrelem)
    tot_weights = sum(quad_rule.weights)
    σ_full = zeros(6)

    εs = [zero(SymmetricTensor{2,2}) for i in 1:nrelem]
    σs = [zero(SymmetricTensor{2,2}) for i in 1:nrelem]
    ε_ps = [zero(SymmetricTensor{2,2}) for i in 1:nrelem]
    for i in 1:nrelem
        for q_point in 1:length(JuAFEM.points(quad_rule))
            w = quad_rule.weights[q_point] / tot_weights
            σs[i] += mss[q_point, i].σ * w
            εs[i] += mss[q_point, i].ε * w
            ε_ps[i] += mss[q_point, i].ε_p * w
            for α = 1:nslips
                τs_di[α, i] += mss[q_point, i].τ_di[α] * w
                τs[α, i] += mss[q_point, i].τ[α] * w
                γs[α, i] += mss[q_point, i].γ[α] * w
                χs[α, i] += mss[q_point, i].χ[α] * w
            end
        end
    end

    dofs_u = mesh.dof[1:2, :]

    disp = u[dofs_u]
    disp = reshape(disp, (2, nnodes))
    disp = [disp; zeros(nnodes)']

    for i in 1:nslips
        dofs_g = mesh.dof[2+i, :]
        grad = u[dofs_g]
        vtk_point_data(vtkfile, grad, "xi_perp_$i")
    end

    vtk_point_data(vtkfile, disp, "displacement")
    vtk_cell_data(vtkfile, τs, "Schmid stress")
    vtk_cell_data(vtkfile, τs_di, "Tau dissip")
    vtk_cell_data(vtkfile, γs, "Slip")
    vtk_cell_data(vtkfile, χs, "div g")
    vtk_cell_data(vtkfile, reinterpret(Float64, εs, (3, length(εs))) ,"Strain")
    vtk_cell_data(vtkfile, reinterpret(Float64, σs, (3, length(σs))), "Stress")
    vtk_cell_data(vtkfile, reinterpret(Float64, ε_ps, (3, length(ε_ps))), "Plastic strain")
    collection_add_timestep(pvd, vtkfile, time)
end


function total_slip{T}(mesh, u::Vector{T}, mss, quad_rule, function_space, nslip, mp)
    fev = FEValues(Float64, quad_rule, function_space)
    γs = Vector{Vector{T}}(nslip)
    tot_slip = 0.0
    tot_grad_en = 0.0
    ndim = 2
    ngradvars = 1
    n_basefuncs = n_basefunctions(get_functionspace(fev))
    nnodes = n_basefuncs
    ed = extract(mesh.edof, u)

    ξ⟂ = Vector{Vector{T}}(nslip)

    for i in 1:nelems(mesh)
        ug = ed[:, i]
        x = [mesh.ex[:, i] mesh.ey[:, i]]'
        x_vec = reinterpret(Vec{2, T}, x, (nnodes,))
        reinit!(fev, x_vec)

        for α in 1:nslip
            ξd = g_dofs(ndim, nnodes, ngradvars, nslip, α)
            ξ⟂[α] = reinterpret(T, ug[ξd], (n_basefuncs,))
        end

        for q_point in 1:length(quad_rule.points)
            for α = 1:nslip
                ξ⟂_gp = function_scalar_value(fev, q_point, ξ⟂[α])

                tot_grad_en += 0.5 * 1 / mp.lα^2 * ξ⟂_gp ⋅ mp.H⟂ * detJdV(fev, q_point)
                γg = mss[q_point, i].γ[α]
                tot_slip += abs2(γ * detJdV(fev, q_point))
            end
        end
    end
    println("total nodes: $(size(mesh.edof, 2))")
    println("total_slip = $(sqrt(tot_slip))")
    println("total_grad = $(tot_grad_en)")
end