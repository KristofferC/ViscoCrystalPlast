using ViscoCrystalPlast
using JuAFEM
using ForwardDiff
using ContMechTensors
using TimerOutputs
using DataFrames
using JLD

import ViscoCrystalPlast: GeometryMesh, Dofs, DirichletBoundaryConditions, CrystPlastMP, QuadratureData
import ViscoCrystalPlast: create_mesh, add_dofs, dofs_element, element_coordinates, move_quadrature_data_to_nodes, interpolate_to


function boundary_f_dual(field::Symbol, x, t::Float64)
    if field == :u
        return 0.01 * x[2] * t
    elseif field == :v
        return 0.01 * x[2] * t
    elseif field == :w
        return 0.0
    else
        error("unhandled field")
    end
end


function startit{dim}(::Dim{dim}, nslips)
    outputt = false

    df = DataFrame(n_elements = Int[], l = Float64[], tot_slip = Float64[], tot_grad_energy = Float64[], tot_elastic_energy = Float64[],
                     err_slip = Float64[], err_grad_energy = Float64[], err_elastic_energy = Float64[])

    df_l_study = DataFrame(l = Float64[], tot_slip = Float64[], tot_grad_energy = Float64[], tot_elastic_en = Float64[])



    function_space = Lagrange{dim, RefTetrahedron, 1}()
    quad_rule = QuadratureRule(Dim{dim}, RefTetrahedron(), 1)
    fe_values = FEValues(Float64, quad_rule, function_space)

    dual_problem = ViscoCrystalPlast.DualProblem(nslips, function_space)

    for l in 0.075
        mp = setup_material(Dim{dim}, l)

        times = linspace(0.0, 10.0, 2)

        ############################
        # Solve fine scale problem #
        ############################
        if dim == 2
            mesh_fine = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/mesh_overkill.mphtxt")
            dofs_fine = ViscoCrystalPlast.add_dofs(mesh_fine, [:u, :v, :ξ⟂1, :ξ⟂2], (2,1,1))
        else
            mesh_fine = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/3d_cube.mphtxt")
            dofs_fine = ViscoCrystalPlast.add_dofs(mesh_fine, [:u, :v, :w, :ξ⟂1, :ξ⟂2, :ξo1, :ξo2], (3,1,1,1,1))
        end

        bcs_fine = ViscoCrystalPlast.DirichletBoundaryConditions(dofs_fine, mesh_fine.boundary_nodes, [:u, :v])
        #pvd_fine = paraview_collection("vtks/shear_dual_fine_$(dim)d")
         pvd_fine = paraview_collection(joinpath(dirname(@__FILE__), "vtks", "shear_dual_fine"))
        timestep_fine = 0
        exporter_fine = (time, u, mss) ->
        begin
            timestep_fine += 1
            mss_nodes = move_quadrature_data_to_nodes(mss, mesh_fine, quad_rule)
            output(pvd_fine, time, timestep_fine, "shear_dual_fine", mesh_fine, dofs_fine, u, mss_nodes, quad_rule, mp)
        end

        sol_fine, mss_fine = ViscoCrystalPlast.solve_problem(dual_problem, mesh_fine, dofs_fine, bcs_fine, fe_values, mp, times,
                                        boundary_f_dual, exporter_fine)
        vtk_save(pvd_fine)


        tot_slip, tot_grad_en, tot_elastic_en = total_slip(mesh_fine, dofs_fine, sol_fine, mss_fine, fe_values, 2, mp)
        push!(df_l_study, [l tot_slip tot_grad_en tot_elastic_en])
        ###############################
        # Solve coarse scale problems #
        ###############################
        #=
        for i in 1:5
            mesh_coarse = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/test_mesh_$i.mphtxt")
            dofs_coarse = ViscoCrystalPlast.add_dofs(mesh_coarse,[:u, :v, :ξ⟂1, :ξ⟂2])
            bcs_coarse = ViscoCrystalPlast.DirichletBoundaryConditions(dofs_coarse, mesh_coarse.boundary_nodes, [:u, :v])
            pvd_coarse = paraview_collection("vtks/shear_dual_coarse")
            timestep_coarse = 0
            exporter_coarse = (time, u, mss) ->
            begin
               timestep_coarse += 1
               mss_nodes = move_quadrature_data_to_nodes(mss, mesh_coarse, quad_rule)
               #output(pvd_coarse, time, timestep_coarse, "shear_dual_coarse", mesh_coarse, dofs_coarse, u, mss_nodes, quad_rule, mp)
            end

            sol_coarse, mss_coarse = ViscoCrystalPlast.solve_problem(ViscoCrystalPlast.DualProblem(), mesh_coarse, dofs_coarse, bcs_coarse, fe_values, mp, times,
                                                                   boundary_f_dual, exporter_coarse)

            mss_coarse_nodes = move_quadrature_data_to_nodes(mss_coarse, mesh_coarse, quad_rule)
            sol_fine_interp, mss_fine_nodes_interp = interpolate_to(sol_coarse, mss_coarse_nodes, mesh_coarse,
                                                   mesh_fine, dofs_coarse, dofs_fine, function_space)



            pvd_diff = paraview_collection("vtks/shear_dual_diff")
            sol_diff = sol_fine - sol_fine_interp
            mss_diff = similar(mss_fine)
            global_gp_coords = ViscoCrystalPlast.get_global_gauss_point_coordinates(fe_values, mesh_fine)
            bounding_elements = ViscoCrystalPlast.find_bounding_element_to_gps(global_gp_coords, mesh_coarse)

            for i in 1:length(mss_diff)
                mss_diff[i] = mss_fine[i] - mss_coarse[bounding_elements[i]]
                #mss_diff[i] = mss_diff[i] .* mss_diff[i]
            end
            mss_diff_nodes = move_quadrature_data_to_nodes(mss_diff, mesh_fine, quad_rule)

            #output(pvd_diff, 1.0, 1, "shear_dual_diff", mesh_fine, dofs_fine, sol_diff, mss_diff_nodes, quad_rule, mp)
            vtk_save(pvd_diff)

            tot_slip, tot_grad_en, tot_elastic_en = total_slip(mesh_coarse, dofs_coarse, sol_coarse, mss_coarse, fe_values, 2, mp)
            err_tot_slip, err_tot_grad_en, err_tot_elastic_en = total_slip(mesh_fine, dofs_fine, sol_diff, mss_diff, fe_values, 2, mp)
            push!(df, [size(mesh_coarse.topology, 2), l, tot_slip, tot_grad_en, tot_elastic_en, err_tot_slip, err_tot_grad_en, err_tot_elastic_en]')
        end
        =#

    end

    if outputt
        save("dataframes/dual_l_study_$(now()).jld", "df", df_l_study)
        save("dataframes/dual_data_frame_$(now()).jld", "df", df)
    end
    return df
end

#=
function startit()
    reset_timer!()
    #mesh = ViscoCrystalPlast.create_mesh("../test/test_mesh.mphtxt")
   # mesh = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/3d_cube.mphtxt")
    mesh = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/test_mesh_3.mphtxt")
    mp = setup_material(Dim{2})

    dofs = ViscoCrystalPlast.add_dofs(mesh, [:u, :v, :ξ⟂1, :ξ⟂2])

    bcs = ViscoCrystalPlast.DirichletBoundaryConditions(dofs, mesh.boundary_nodes, [:u, :v])

    function_space = JuAFEM.Lagrange{2, RefTetrahedron, 1}()
    quad_rule = QuadratureRule(Dim{2}, RefTetrahedron(), 1)
    fe_values = FEValues(Float64, quad_rule, function_space)

    times = linspace(0.0, 100.0, 10)
    pvd = paraview_collection("vtks/shear_dual")
    exporter = (time, u, mss) -> output(pvd, time, mesh, dofs, u, mss, quad_rule, mp)
    ViscoCrystalPlast.solve_problem(ViscoCrystalPlast.DualProblem(), mesh, dofs, bcs, fe_values, mp, times, boundary_f, exporter)
    vtk_save(pvd)
    print_timer()
end
=#

function setup_material{dim}(::Type{Dim{dim}}, lα)
    E = 200000.0
    ν = 0.3
    n = 2.0
    #lα = 0.5
    H⟂ = 0.1E
    Ho = 0.1E
    C = 1.0e3
    tstar = 1000.0
    angles = [20.0, 40.0]
    mp = ViscoCrystalPlast.CrystPlastMP(Dim{dim}, E, ν, n, H⟂, Ho, lα, tstar, C, angles)
    return mp
end

function output{QD <: QuadratureData, dim}(pvd, time, timestep, filename, mesh, dofs, u,
                                           mss_nodes::AbstractVector{QD}, quad_rule::QuadratureRule{dim}, mp)
    nslips = length(mp.angles)
    nodes_per_ele = dim == 2 ? 3 : 4
    n_sym_components = dim == 2 ? 3 : 6
    tot_nodes = size(mesh.coords, 2)
    nrelem = size(mesh.topology, 2)
    #vtkfile = vtk_grid(mesh.topology, mesh.coords, "vtks/" * filename * "_$timestep")
    vtkfile = vtk_grid(mesh.topology, mesh.coords, joinpath(dirname(@__FILE__), "vtks", "$filename" * "_$timestep"))

    vtk_point_data(vtkfile, reinterpret(Float64, SymmetricTensor{2, dim, Float64, 3}[mss_nodes[i].σ for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Stress")
    vtk_point_data(vtkfile, reinterpret(Float64, SymmetricTensor{2, dim, Float64, 3}[mss_nodes[i].ε  for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Strain")
    vtk_point_data(vtkfile, reinterpret(Float64, SymmetricTensor{2, dim, Float64, 3}[mss_nodes[i].ε_p for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Plastic strain")

    for α in 1:nslips
        vtk_point_data(vtkfile, Float64[mss_nodes[i].γ[α] for i in 1:tot_nodes], "Slip $α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].τ[α] for i in 1:tot_nodes], "Schmid $α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].τ_di[α] for i in 1:tot_nodes], "Tau dissip $α")
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
        vtk_point_data(vtkfile, grad, "xi_o_$i")
    end

    vtk_point_data(vtkfile, disp, "displacement")

    collection_add_timestep(pvd, vtkfile, time)

end

function total_slip{T, dim}(mesh, dofs, u::Vector{T}, mss, fev::FEValues{dim}, nslip, mp)
    ξ⟂s = Vector{Vector{T}}(nslip)
    ξos = Vector{Vector{T}}(nslip)

    tot_slip = 0.0
    tot_grad_en = 0.0
    tot_elastic_en = 0.0
    ngradvars = dim - 1
    n_basefuncs = n_basefunctions(get_functionspace(fev))
    nnodes = n_basefuncs

    e_coordinates = zeros(dim, n_basefuncs)

    for i in 1:size(mesh.topology, 2)
        ViscoCrystalPlast.element_coordinates!(e_coordinates , mesh, i)
        edof = ViscoCrystalPlast.dofs_element(mesh, dofs, i)
        ug = u[edof]
        x_vec = reinterpret(Vec{dim, T}, e_coordinates, (nnodes,))
        reinit!(fev, x_vec)
        for α in 1:nslip
            if dim == 2
                ξ⟂_node_dofs = ViscoCrystalPlast.compute_γdofs(dim, nnodes, ngradvars, nslip, α)
            else
                ξ⟂_node_dofs = ξ_dofs(dim, nnodes, ngradvars, nslip, α, :ξ⟂)
                ξo_node_dofs = ξ_dofs(dim, nnodes, ngradvars, nslip, α, :ξo)
            end
            ξ⟂s[α] = ug[ξ⟂_node_dofs]
            if dim == 3
                ξos[α] = ug[ξo_node_dofs]
            end
        end

        for q_point in 1:length(points(get_quadrule(fev)))
            σ = mss[q_point, i].σ
            ε = mss[q_point, i].ε
            tot_elastic_en += 0.5 * ε ⊡ σ * detJdV(fev, q_point)
            for α = 1:nslip
                ξ⟂ = function_scalar_value(fev, q_point, ξ⟂s[α])
                if dim == 3
                    ξo = function_scalar_value(fev, q_point, ξos[α])
                end

                if dim == 2
                    tot_grad_en += 0.5 / mp.lα^2 * (ξ⟂^2 / mp.H⟂) * detJdV(fev, q_point)
                else
                    tot_grad_en += 0.5 / mp.lα^2 * (ξo^2 / mp.Ho + ξ⟂^2 / mp.H⟂) * detJdV(fev, q_point)
                end
                tot_slip += mss[q_point, i].γ[α]^2 * detJdV(fev, q_point)
            end
        end
    end
    #println("total nodes: $(size(mesh.edof, 2))")
    println("effective  slip = $(sqrt(tot_slip))")
    println("total_grad = $(tot_grad_en)")

    return sqrt(tot_slip), tot_grad_en, tot_elastic_en
end

