using ViscoCrystalPlast
using JuAFEM
using ForwardDiff
using ContMechTensors
using TimerOutputs
using DataFrames
using JLD


import ViscoCrystalPlast: GeometryMesh, Dofs, DirichletBoundaryConditions, CrystPlastMP, QuadratureData
import ViscoCrystalPlast: create_mesh, add_dofs, dofs_element, element_coordinates, move_quadrature_data_to_nodes, interpolate_to

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


function startit{dim}(::Type{Dim{dim}})


    df = DataFrame(n_elements = Int[], l = Float64[], tot_slip = Float64[], tot_grad_energy = Float64[], tot_elastic_energy = Float64[],
                     err_slip = Float64[], err_grad_energy = Float64[], err_elastic_energy = Float64[])

    df_l_study = DataFrame(l = Float64[], tot_slip = Float64[], tot_grad_energy = Float64[], tot_elast_en = Float64[])

    function_space = Lagrange{dim, RefTetrahedron, 1}()
    quad_rule = QuadratureRule(Dim{dim}, RefTetrahedron(), 1)
    fe_values = FEValues(Float64, quad_rule, function_space)

    primal_problem = ViscoCrystalPlast.PrimalProblem(dim, function_space)

    write_dataframe = false

    for l in 0.1:0.1:0.1
        mp = setup_material(Dim{dim}, l)

        times = linspace(0.0, 10.0, 2)

        ############################
        # Solve fine scale problem #
        ############################
        mesh_fine = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/mesh_overkill.mphtxt")
        if dim == 2
            mesh_fine = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/mesh_overkill.mphtxt")
            dofs_fine = ViscoCrystalPlast.add_dofs(mesh_fine, [:u, :v, :γ1, :γ2], (2, 1, 1))
            bcs_fine = ViscoCrystalPlast.DirichletBoundaryConditions(dofs_fine, mesh_fine.boundary_nodes, [:u, :v, :γ1, :γ2])
        else
            mesh_fine = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/3d_cube.mphtxt")
            dofs_fine = ViscoCrystalPlast.add_dofs(mesh_fine, [:u, :v, :w, :γ1, :γ2], (3, 1, 1))
            bcs_fine = ViscoCrystalPlast.DirichletBoundaryConditions(dofs_fine, mesh_fine.boundary_nodes, [:u, :v, :w, :γ1, :γ2])
        end

        pvd_fine = paraview_collection(joinpath(dirname(@__FILE__), "vtks", "shear_primal_fine"))
        timestep_fine = 0
        exporter_fine = (time, u, mss) ->
        begin
            timestep_fine += 1
            mss_nodes = move_quadrature_data_to_nodes(mss, mesh_fine, quad_rule)
            output(pvd_fine, time, timestep_fine, "shear_primal_fine", mesh_fine, dofs_fine, u, mss_nodes, quad_rule, mp)
        end

        sol_fine, mss_fine = ViscoCrystalPlast.solve_problem(primal_problem, mesh_fine, dofs_fine, bcs_fine, fe_values, mp, times,
                                        boundary_f_primal, exporter_fine)
        vtk_save(pvd_fine)

       # tot_slip, tot_grad_en, tot_elastic_en = total_slip(mesh_fine, dofs_fine, sol_fine, mss_fine, fe_values, 2, mp)
        #push!(df_l_study, [l tot_slip tot_grad_en tot_elastic_en])

        ###############################
        # Solve coarse scale problems #
        ###############################
        #=
        for i in 1:5
            mesh_coarse = ViscoCrystalPlast.create_mesh("/home/kristoffer/Dropbox/PhD/Research/CrystPlast/meshes/test_mesh_$i.mphtxt")
            dofs_coarse = ViscoCrystalPlast.add_dofs(mesh_coarse, [:u, :v, :γ1, :γ2])
            bcs_coarse = ViscoCrystalPlast.DirichletBoundaryConditions(dofs_coarse, mesh_coarse.boundary_nodes, [:u, :v, :γ1, :γ2])
            pvd_coarse = paraview_collection("vtks/shear_primal_coarse")
            timestep_coarse = 0
            exporter_coarse = (time, u, mss) ->
            begin
               timestep_coarse += 1
               mss_nodes = move_quadrature_data_to_nodes(mss, mesh_coarse, quad_rule)
               #output(pvd_coarse, time, timestep_coarse, "shear_primal_coarse", mesh_coarse, dofs_coarse, u, mss_nodes, quad_rule, mp)
            end

            sol_coarse, mss_coarse = ViscoCrystalPlast.solve_problem(ViscoCrystalPlast.PrimalProblem(), mesh_coarse, dofs_coarse, bcs_coarse, fe_values, mp, times,
                                                                   boundary_f_primal, exporter_coarse)

            mss_coarse_nodes = move_quadrature_data_to_nodes(mss_coarse, mesh_coarse, quad_rule)
            sol_fine_interp, mss_fine_nodes_interp = interpolate_to(sol_coarse, mss_coarse_nodes, mesh_coarse,
                                                   mesh_fine, dofs_coarse, dofs_fine, function_space)


            pvd_diff = paraview_collection("vtks/shear_primal_diff")
            sol_diff = sol_fine - sol_fine_interp
            mss_diff = similar(mss_fine)
            global_gp_coords = ViscoCrystalPlast.get_global_gauss_point_coordinates(fe_values, mesh_fine)
            bounding_elements = ViscoCrystalPlast.find_bounding_element_to_gps(global_gp_coords, mesh_coarse)

            for i in 1:length(mss_diff)
                mss_diff[i] = mss_fine[i] - mss_coarse[bounding_elements[i]]
                #mss_diff[i] = mss_diff[i] .* mss_diff[i]
            end
            mss_diff_nodes = move_quadrature_data_to_nodes(mss_diff, mesh_fine, quad_rule)

            #output(pvd_diff, 1.0, 1, "shear_primal_diff", mesh_fine, dofs_fine, sol_diff, mss_diff_nodes, quad_rule, mp)
            vtk_save(pvd_diff)

            tot_slip, tot_grad_en, tot_elastic_en = total_slip(mesh_coarse, dofs_coarse, sol_coarse, mss_coarse, fe_values, 2, mp)
            err_tot_slip, err_tot_grad_en, err_tot_elastic_en = total_slip(mesh_fine, dofs_fine, sol_diff, mss_diff, fe_values, 2, mp)
            push!(df, [size(mesh_coarse.topology, 2), l, tot_slip, tot_grad_en, tot_elastic_en, err_tot_slip, err_tot_grad_en, err_tot_elastic_en]')
        end
 =#
    end


    if write_dataframe
        save("dataframes/primal_l_study_$(now()).jld", "df", df_l_study)

        save("dataframes/primal_data_frame_$(now()).jld", "df", df)
    end
    return df
end

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
    vtkfile = vtk_grid(mesh.topology, mesh.coords, joinpath(dirname(@__FILE__), "vtks", "$filename" * "_$timestep"))

    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].σ for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Stress")
    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].ε  for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Strain")
    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].ε_p for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Plastic strain")
    for α in 1:nslips
        vtk_point_data(vtkfile, Float64[mss_nodes[i].τ[α] for i in 1:tot_nodes], "Schmid $α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].τ_di[α] for i in 1:tot_nodes], "Tau dissip $α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].ξo[α] for i in 1:tot_nodes], "xi o $α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].ξ⟂[α] for i in 1:tot_nodes], "xi perp $α")
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
    vtk_save(vtkfile)

    collection_add_timestep(pvd, vtkfile, time)

end

function total_slip{T}(mesh, dofs, u::Vector{T}, mss, fev, nslip, mp)
    γs = Vector{Vector{T}}(nslip)
    tot_slip = 0.0
    tot_grad_en = 0.0
    tot_elastic_en = 0.0
    ngradvars = 1
    n_basefuncs = n_basefunctions(get_functionspace(fev))
    nnodes = n_basefuncs

    e_coordinates = zeros(dim, n_basefuncs)

    for i in 1:size(mesh.topology, 2)
        ViscoCrystalPlast.element_coordinates!(e_coordinates , mesh, i)
        edof = ViscoCrystalPlast.dofs_element(mesh, dofs, i)
        ug = u[edof]
        x_vec = reinterpret(Vec{2, T}, e_coordinates, (nnodes,))
        reinit!(fev, x_vec)
        for α in 1:nslip
            gd = ViscoCrystalPlast.γ_dofs(dim, nnodes, ngradvars, nslip, α)
            γs[α] = ug[gd]
        end

        for q_point in 1:length(points(get_quadrule(fev)))
            σ = mss[q_point, i].σ
            ε = mss[q_point, i].ε
            tot_elastic_en += 0.5 * ε ⊡ σ * detJdV(fev, q_point)
            for α = 1:nslip
                ξo = mss[q_point, i].ξo[α]
                ξ⟂ = mss[q_point, i].ξ⟂[α]
                #println(ξo, " ", ξ⟂)
                tot_grad_en += 0.5 / mp.lα^2 * (ξ⟂^2 / mp.H⟂ + ξo^2 / mp.Ho) * detJdV(fev, q_point)
                γ = function_scalar_value(fev, q_point, γs[α])
                tot_slip += γ^2 * detJdV(fev, q_point)
            end
        end
    end
    #println("total nodes: $(size(mesh.edof, 2))")
    #println("effective  slip = $(sqrt(tot_slip))")
    #println("total_grad = $(tot_grad_en)")

    return sqrt(tot_slip), tot_grad_en, tot_elastic_en
end

#startit()
