include("quadrature_data.jl")
include("intf_primal.jl")




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
=#

#=
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