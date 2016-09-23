using ViscoCrystalPlast
using JuAFEM
using ForwardDiff
using ContMechTensors
using TimerOutputs
using DataFrames
using JLD
using FileIO

import ViscoCrystalPlast: GeometryMesh, Dofs, DirichletBoundaryConditions, CrystPlastMP, QuadratureData, DofHandler
import ViscoCrystalPlast: add_dofs, dofs_element, element_coordinates, move_quadrature_data_to_nodes, interpolate_to
import ViscoCrystalPlast: element_set, node_set, element_vertices

import ViscoCrystalPlast: add_field!, close!, element_set, element_coordinates,
                            add_dirichletbc!, ndim, ndofs, nelements, nnodes, free_dofs, update_dirichletbcs!, apply!,
                                create_lookup, add_element_set!, add_node_set!


function startit{dim}(::Type{Dim{dim}}, nslips, probtype)
    for probtype in (:primal, :dual)
        function_space = Lagrange{dim, RefTetrahedron, 1}()
        quad_rule = QuadratureRule(Dim{dim}, RefTetrahedron(), 1)
        fe_values = FEValues(Float64, quad_rule, function_space)

        files = readdir(joinpath(dirname(@__FILE__), "..", "meshes"))
        rcls_matches = [match(r"n3_rcl.(.*?).inp", f)  for  f in files]
        rcls = [rcl_match.captures[1] for rcl_match in rcls_matches[rcls_matches .!= nothing]]
        for rcl in reverse(rcls)
            for alpha in (40.0, 41.0, 42.0, 43.0, 44.0, 45.0)
                ########################
                # Set up mesh and dofs #
                ########################
                f = joinpath(dirname(@__FILE__), "../meshes/n3_rcl.$rcl.inp")

                # slip_boundary is created below
                mesh, dh = ViscoCrystalPlast.create_mesh_and_dofhandler(f, dim, nslips, probtype)
                ViscoCrystalPlast.add_node_set!(mesh, "RVE_boundary", ViscoCrystalPlast.get_RVE_boundary_nodes(mesh))


                dbcs = ViscoCrystalPlast.DirichletBoundaryConditions(dh)
                for i in 1:nslips
                    if probtype == :primal
                        ViscoCrystalPlast.add_dirichletbc!(dbcs, Symbol("slip_", i), ViscoCrystalPlast.node_set(mesh, "RVE_boundary"), (x,t) -> 0.0)
                        ViscoCrystalPlast.add_dirichletbc!(dbcs, Symbol("slip_", i), ViscoCrystalPlast.node_set(mesh, "slip_boundary"), (x,t) -> 0.0)
                        # ViscoCrystalPlast.add_dirichletbc!(dbcs, Symbol("γ", i), ViscoCrystalPlast.node_set(mesh, "RVE_boundary"), (x,t) -> 0.0)
                    else
                        #ViscoCrystalPlast.add_dirichletbc!(dbcs, Symbol("xi_perp_", i), ViscoCrystalPlast.node_set(mesh, "RVE_boundary"), (x,t) -> 0.0)
                        if dim == 3
                        #    ViscoCrystalPlast.add_dirichletbc!(dbcs, Symbol("xi_o_", i), ViscoCrystalPlast.node_set(mesh, "RVE_boundary"), (x,t) -> 0.0)
                        end
                    end
                end

                ViscoCrystalPlast.add_node_set!(mesh, "RVE_boundary", ViscoCrystalPlast.get_RVE_boundary_nodes(mesh))
                if dim == 2
                    ViscoCrystalPlast.add_dirichletbc!(dbcs, :u, ViscoCrystalPlast.node_set(mesh, "RVE_boundary"), (x,t) -> t * 0.01 * [x[1], 0.0], collect(1:dim))
                else
                    ViscoCrystalPlast.add_dirichletbc!(dbcs, :u, ViscoCrystalPlast.node_set(mesh, "RVE_boundary"), (x,t) -> t * 0.01 * [x[1], x[2], 2*x[3]], collect(1:dim))
                end
                ViscoCrystalPlast.close!(dbcs)

                polys = ViscoCrystalPlast.create_element_to_grain_map(mesh)

                for l in 0.35
                    if dim == 2
                        mps = [setup_material(Dim{2}, l, nslips, alpha) for i in 1:length(unique(polys))]
                    else
                        mps = [setup_material_3d(Dim{3}, l, nslips) for i in 1:length(unique(polys))]
                    end
                    # Regenerate the problems to reset the quadrature data
                    if probtype == :primal
                        problem = ViscoCrystalPlast.PrimalProblem(nslips, function_space)
                    elseif probtype == :dual
                        problem = ViscoCrystalPlast.DualProblem(nslips, function_space)
                    else
                        error("Invalid problem type")
                    end

                    times = linspace(0.0, 10.0, 10)
                    ############################
                    # Solve fine scale problem #
                    ############################
                    pvd_fine = paraview_collection(joinpath(dirname(@__FILE__), "vtks", "shear" * string(probtype) * "_fine_$(dim)d_$l"))
                    timestep_fine = 0
                    exporter_fine = (time, u, mss) ->
                    begin
                        timestep_fine += 1
                        mss_nodes = ViscoCrystalPlast.move_quadrature_data_to_nodes(mss, mesh, quad_rule)
                        output(problem, pvd_fine, time, timestep_fine, "grain_dir_exp_$(probtype)_$(rcl)_$(alpha)" * string(probtype), mesh, dh, u, dbcs, mss_nodes, quad_rule, mps, polys)
                    end

                    sol, mss = ViscoCrystalPlast.solve_problem(problem, mesh, dh, dbcs, fe_values, mps, times,
                                                                          exporter_fine, polys)
                    #vtk_save(pvd_fine)

                    save(joinpath(dirname(@__FILE__), "raw_data", "grain_dir_exp_$(probtype)_$(rcl)_$(alpha).jld"),
                        "function_space", function_space, "quad_rule", quad_rule, "mesh" ,mesh, "dh", dh,
                        "sol", sol, "mss", mss, "mps", mps)

                    #load(joinpath(dirname(@__FILE__), "raw_data", "$(probtype)_$(rcl)_$(alpha).jld"))
                end
            end
        end
    end
    return
end

function setup_material{dim}(::Type{Dim{dim}}, lα::Float64, nslips::Int, angle)
    E = 200000.0
    ν = 0.3
    n = 2.0
    #lα = 0.5
    H⟂ = 0.1E
    Ho = 0.1E
    C = 1.0e3
    tstar = 1000.0
    #angles = [20.0, 40.0]
    angles = [angle]
    srand(12345)
    #angles = [90 * rand(), 90 * rand()]
    @assert length(angles) == nslips
    mp = ViscoCrystalPlast.CrystPlastMP(Dim{dim}, E, ν, n, H⟂, Ho, lα, tstar, C, angles)
    return mp
end


function rand_eul()
    α = 2*(rand() - 0.5) * π
    γ = 2*(rand() - 0.5) * π
    β = rand() * π
    return (α, γ, β)
end

function setup_material_3d{dim}(::Type{Dim{dim}}, lα, nslips::Int)
    srand(1234)
    E = 200000.0
    ν = 0.3
    n = 2.0
    lα = 0.5
    H⟂ = 0.1E
    Ho = 0.1E
    C = 1.0e3
    tstar = 1000.0
    ϕs = [rand_eul() for i in 1:nslips]
    @assert length(ϕs) == nslips
    mp = ViscoCrystalPlast.CrystPlastMP(Dim{dim}, E, ν, n, H⟂, Ho, lα, tstar, C, ϕs)
    return mp
end

function output{QD <: ViscoCrystalPlast.QuadratureData, dim}(::ViscoCrystalPlast.PrimalProblem, pvd, time, timestep, filename, mesh, dh, u, dbcs,
                                           mss_nodes::AbstractVector{QD}, quad_rule::QuadratureRule{dim}, mps, polys)
    nodes_per_ele = dim == 2 ? 3 : 4
    n_sym_components = dim == 2 ? 3 : 6
    tot_nodes = nnodes(mesh)
    mp = mps[1]
    vtkfile = vtk_grid(mesh, joinpath(dirname(@__FILE__), "vtks", "$filename" * "_$timestep"), compress=true, append=true)


    vtk_point_data(vtkfile, dh, u)
    vtk_point_data(vtkfile, dbcs)

    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].σ for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Stress")
    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].ε  for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Strain")
    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].ε_p for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Plastic strain")
    for α in 1:length(mp.angles)
        vtk_point_data(vtkfile, Float64[mss_nodes[i].τ[α] for i in 1:tot_nodes], "Schmid $α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].τ_di[α] for i in 1:tot_nodes], "Tau dissip $α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].ξo[α] for i in 1:tot_nodes], "xi o $α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].ξ⟂[α] for i in 1:tot_nodes], "xi perp $α")
    end

    vtk_cell_data(vtkfile, convert(Vector{Float64}, polys), "poly")
    vtk_save(vtkfile)
    #collection_add_timestep(pvd, vtkfile, time)

end


function output{QD <: QuadratureData, dim}(::ViscoCrystalPlast.DualProblem, pvd, time, timestep, filename, mesh, dh, u, dbcs,
                                           mss_nodes::AbstractVector{QD}, quad_rule::QuadratureRule{dim}, mps, polys)

   nodes_per_ele = dim == 2 ? 3 : 4
   n_sym_components = dim == 2 ? 3 : 6
   tot_nodes = nnodes(mesh)
   mp = mps[1]
    #vtkfile = vtk_grid(mesh.topology, mesh.coords, "vtks/" * filename * "_$timestep")
    vtkfile = vtk_grid(mesh, joinpath(dirname(@__FILE__), "vtks", "$filename" * "_$timestep"))

    vtk_point_data(vtkfile, dh, u)
    vtk_point_data(vtkfile, dbcs)

    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].σ for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Stress")
    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].ε  for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Strain")
    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].ε_p for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Plastic strain")
    for α in 1:length(mp.angles)
        vtk_point_data(vtkfile, Float64[mss_nodes[i].γ[α] for i in 1:tot_nodes], "Slip $α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].τ[α] for i in 1:tot_nodes], "Schmid $α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].τ_di[α] for i in 1:tot_nodes], "Tau dissip $α")
    end
    vtk_cell_data(vtkfile, convert(Vector{Float64}, polys), "poly")
    vtk_save(vtkfile)
end
