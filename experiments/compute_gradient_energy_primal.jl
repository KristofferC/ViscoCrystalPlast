using ViscoCrystalPlast
using JuAFEM
using ForwardDiff
using ContMechTensors
using TimerOutputs


import ViscoCrystalPlast: GeometryMesh, Dofs, DirichletBoundaryConditions, CrystPlastMP
import ViscoCrystalPlast: create_mesh, add_dofs, dofs_element, element_coordinates

function boundary_f_primal(field::Symbol, x, t::Float64)
    if field == :u
        return 0.01 * x[2] * t
    elseif field == :v
        return 0.01 * x[2] * t
    elseif field == :w
        return 0.0
    elseif startswith(string(field), "γ")
        return 0.0
    else
        error("unhandled field")
    end
end

function startit()
    reset_timer!()
    #mesh = ViscoCrystalPlast.create_mesh("../test/test_mesh.mphtxt")
    #mesh = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/3d_cube.mphtxt")
    mesh = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/test_mesh_3.mphtxt")
    mp = setup_material(Dim{2})


    dofs = ViscoCrystalPlast.add_dofs(mesh, [:u, :v, :γ1, :γ2])

    bcs = ViscoCrystalPlast.DirichletBoundaryConditions(dofs, mesh.boundary_nodes, [:u, :v, :γ1, :γ2])

    function_space = Lagrange{2, RefTetrahedron, 1}()
    quad_rule = QuadratureRule(Dim{2}, RefTetrahedron(), 1)
    fe_values = FEValues(Float64, quad_rule, function_space)

    times = linspace(0.0, 100.0, 10)

    pvd = paraview_collection("vtks/shear_primal")
    exporter = (time, u, mss) -> output(pvd, time, mesh, dofs, u, mss, quad_rule, mp)
    ViscoCrystalPlast.solve_problem(ViscoCrystalPlast.PrimalProblem(), mesh, dofs, bcs, fe_values, mp, times,
                                    boundary_f_primal, exporter)
    vtk_save(pvd)

    print_timer()
end

function setup_material{dim}(::Type{Dim{dim}})
    E = 200000.0
    ν = 0.3
    n = 2.0
    lα = 0.5
    H⟂ = 0.1E
    Ho = 0.1E
    C = 1.0e3
    tstar = 1000.0
    angles = [20.0, 40.0]
    mp = ViscoCrystalPlast.CrystPlastMP(Dim{dim}, E, ν, n, H⟂, Ho, lα, tstar, C, angles)
    return mp
end


timestep = 0
function output{dim}(pvd, time, mesh, dofs, u, mss, quad_rule::QuadratureRule{dim}, mp)
    global timestep
    timestep += 1
    nslips = length(mp.angles)
    nodes_per_ele = dim == 2 ? 3 : 4
    n_sym_components = dim == 2 ? 3 : 6
    tot_nodes = size(mesh.coords, 2)
    nrelem = size(mesh.topology, 2)
    vtkfile = vtk_grid(mesh.topology, mesh.coords, "vtks/box_primal_$timestep")


    mss_nodes = [ViscoCrystalPlast.CrystPlastPrimalQD(nslips, Dim{dim}) for i = 1:tot_nodes]
    count_nodes = zeros(Int, tot_nodes)
    for i in 1:nrelem
        for q_point in 1:length(points(quad_rule))
            for node in mesh.topology[:, i]
                count_nodes[node] += 1
                mss_nodes[node].σ += mss[q_point, i].σ
                mss_nodes[node].ε += mss[q_point, i].ε
                mss_nodes[node].ε_p += mss[q_point, i].ε_p

                for α = 1:nslips
                    mss_nodes[node].ξ⟂[α] += mss[q_point, i].ξ⟂[α]
                    mss_nodes[node].ξo[α] += mss[q_point, i].ξo[α]
                    mss_nodes[node].τ[α] += mss[q_point, i].τ[α]
                    mss_nodes[node].τ_di[α] += mss[q_point, i].τ_di[α]
                end
            end
        end
    end

     for i in 1:tot_nodes
        mss_nodes[i].σ /= count_nodes[i]
        mss_nodes[i].ε /= count_nodes[i]
        mss_nodes[i].ε_p /= count_nodes[i]
        for α = 1:nslips
            mss_nodes[i].τ_di[α] /= count_nodes[i]
            mss_nodes[i].τ[α] /= count_nodes[i]
            mss_nodes[i].ξ⟂[α] /= count_nodes[i]
            mss_nodes[i].ξo[α]  /= count_nodes[i]
        end
    end

    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].σ for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Stress")
    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].ε  for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Strain")
    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].ε_p for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Plastic strain")
    for α in 1:nslips
        vtk_point_data(vtkfile, [mss_nodes[i].τ[α] for i in 1:tot_nodes], "Schmid $α")
        vtk_point_data(vtkfile, [mss_nodes[i].τ_di[α] for i in 1:tot_nodes], "Tau dissip $α")
        vtk_point_data(vtkfile, [mss_nodes[i].ξo[α] for i in 1:tot_nodes], "xi o $α")
        vtk_point_data(vtkfile, [mss_nodes[i].ξ⟂[α] for i in 1:tot_nodes], "xi perp $α")
    end


    dofs_u = dofs.dof_ids[1:dim, :]

    disp = u[dofs_u]
    disp = reshape(disp, size(mesh.coords))
    if dim == 2
        disp = [disp; zeros(size(mesh.coords,2))']
    end

    for i in 1:nslips
        dofs_g = dofs.dof_ids[dim+i, :]
        grad = u[dofs_g]
        vtk_point_data(vtkfile, grad, "slip_$i")
    end

    vtk_point_data(vtkfile, disp, "displacement")

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





startit()


