using ViscoCrystalPlast
using JuAFEM
using Quaternions

#using ForwardDiff
using Tensors
using TimerOutputs
using DataFrames
using JLD
using FileIO



import ViscoCrystalPlast: CrystPlastMP, QuadratureData
import ViscoCrystalPlast: move_quadrature_data_to_nodes
import ViscoCrystalPlast.Dim

import ViscoCrystalPlast: ndim, ndofs


function setup_material_3d(lα, nslips::Int)
    #srand(1234)
    E = 200000.0
    ν = 0.3
    n = 8.0
    H⟂ = 0.00001E
    Ho = 0.00001E
    C = 1.0e3
    tstar = 1000.0
    rotmats = [rotationmatrix(quatrand()) for i in 1:nslips]
    mp = ViscoCrystalPlast.CrystPlastMP(Dim{3}, E, ν, n, H⟂, Ho, lα, tstar, C, rotmats)
    return mp
end

function setup_rand_BCC(lα)
    s1 = Vec{3}((1., 1.,-1.)) / √3
    s2 = Vec{3}((1.,-1.,-1.)) / √3
    s3 = Vec{3}((1.,-1., 1.)) / √3
    s4 = Vec{3}((1., 1., 1.)) / √3

    slip_planes = [
    s1, s1, s1,
    s2, s2, s2,
    s3, s3, s3,
    s4, s4, s4,
    ]

    slip_directions = [
    Vec{3}((0., 1., 1.)) / √2,
    Vec{3}((1., 0., 1.)) / √2,
    Vec{3}((1.,-1., 0.)) / √2,
    Vec{3}((0., 1.,-1.)) / √2,
    Vec{3}((1., 0., 1.)) / √2,
    Vec{3}((1., 1., 0.)) / √2,
    Vec{3}((0., 1., 1.)) / √2,
    Vec{3}((1., 0.,-1.)) / √2,
    Vec{3}((1., 1., 0.)) / √2,
    Vec{3}((0., 1.,-1.)) / √2,
    Vec{3}((1., 0.,-1.)) / √2,
    Vec{3}((1.,-1., 0.)) / √2,
    ]

    z = 2*(rand() - 0.5)
    ϕ = 2π*rand()
    x = sqrt(1 - z^2) * cos(ϕ)
    y = sqrt(1 - z^2) * sin(ϕ)

    u = Vec{3}((x, y, z))
    θ = 2π*rand()


    for i in 1:length(slip_planes)
        slip_planes[i] = rotate(slip_planes[i], u, θ)
        slip_directions[i] = rotate(slip_directions[i], u, θ)
    end


    E = 200000.0
    ν = 0.3
    n = 4.0
    H⟂ = 0.00001E
    Ho = 0.00001E
    C = 1.0e3
    tstar = 1000.0
    mp = ViscoCrystalPlast.CrystPlastMP(Dim{3}, E, ν, n, H⟂, Ho, lα, tstar, C, slip_directions, slip_planes)
    return mp
end

to_voigt(s::SymmetricTensor{2, 2}) = (s[1,1], s[2,2], s[1,2])
to_voigt(s::SymmetricTensor{2, 3}) = (s[1,1], s[2,2], s[3,3], s[1,2], s[2,3], s[1,3])


function output{QD <: QuadratureData, dim}(::ViscoCrystalPlast.DualProblem, pvd, time, timestep, filename, mesh, dh, u, dbcs,
                                           mss_nodes::AbstractVector{QD}, quad_rule::QuadratureRule{dim}, mps, polys)

    nodes_per_ele = dim == 2 ? 3 : 4
    n_sym_components = dim == 2 ? 3 : 6
    tot_nodes = getnnodes(mesh)
    mp = mps[1]
    #vtkfile = vtk_grid(mesh.topology, mesh.coords, "vtks/" * filename * "_$timestep")
    vtkfile = vtk_grid(joinpath(dirname(@__FILE__), "vtks", "$filename" * "_$timestep"), mesh)

    vtk_point_data(vtkfile, dh, u)
    vtk_point_data(vtkfile, dbcs)

    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].σ for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Stress")
    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].ε  for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Strain")
    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].ε_p for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Plastic strain")
    for α in 1:length(mp.s)
        vtk_point_data(vtkfile, Float64[mss_nodes[i].γ[α] for i in 1:tot_nodes], "Slip $α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].τ[α] for i in 1:tot_nodes], "Schmid $α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].τ_di[α] for i in 1:tot_nodes], "Tau dissip $α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].χ⟂[α] for i in 1:tot_nodes], "Chi_perp_$α")
        vtk_point_data(vtkfile, Float64[mss_nodes[i].χo[α] for i in 1:tot_nodes], "Chi_odot_$α")
    end
    vtk_cell_data(vtkfile, convert(Vector{Float64}, polys), "poly")
    vtk_save(vtkfile)
end

function integrate(f, mesh, cellvalues)
    init = f(1,1)
    init -= init
    Ω = 0.0
    for element_id in 1:getncells(mesh)
        reinit!(cellvalues, getcoordinates(mesh, element_id))
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            Ω += dΩ
            init += f(element_id, q_point) * dΩ
        end
    end
    return init / Ω
end

function integrate_grains(f, mesh, cellvalues, polys)
    init = f(1,1)
    init -= init
    n_polys = length(unique(polys))
    results = [init for i in 1:n_polys]
    Ω_grain = zeros(n_polys)
    for element_id in 1:getncells(mesh)
        reinit!(cellvalues, getcoordinates(mesh, element_id))
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            Ω_grain[polys[element_id]] += dΩ
            results[polys[element_id]] += f(element_id, q_point) * dΩ
        end
    end
    return results ./ Ω_grain, Ω_grain
end


function runit()
    dim = 3
    srand(1234)
    nslips = 12
    bc = :microfree
    probtype = :dual
    function_space = Lagrange{dim, RefTetrahedron, 1}()
    quad_rule = QuadratureRule{dim, RefTetrahedron}(1)
    fev_u = CellVectorValues(Float64, quad_rule, function_space)
    fev_ξ = CellScalarValues(Float64, quad_rule, function_space)
    f = "/home/kristoffer/neperout/n30-id1.inp"

  # slip_boundary is created below
    mesh, dh = ViscoCrystalPlast.create_mesh_and_dofhandler(f, dim, nslips, probtype)
    polys = ViscoCrystalPlast.create_element_to_grain_map(mesh)

    n_polys = length(unique(polys))
    L_poly = 0.1
    V_poly = L_poly^dim
    # V_mesh = L_mesh^dim = V_poly * N_poly
    L_mesh = (V_poly * n_polys)^(1/dim)
    new_nodes = similar(getnodes(mesh))
    for n in 1:length(new_nodes)
        new_nodes[n] = JuAFEM.Node(L_mesh * getnodes(mesh, n).x)
    end
    mesh.nodes = new_nodes

    l = 0.1
    Δt = 1.0
    endt = 15.0
    times = 0.0:Δt:endt
    ɛend = 0.3


    dbcs = ViscoCrystalPlast.DirichletBoundaryConditions(dh)
    JuAFEM.addnodeset!(mesh, "RVE_boundary", ViscoCrystalPlast.get_RVE_boundary_nodes(mesh))

    if dim == 2
        add!(dbcs, :u, getnodeset(mesh, "RVE_boundary"), (x,t) -> t / last(times)  * ɛend * [x[2], 0.0], collect(1:dim))
    else
        add!(dbcs, :u, getnodeset(mesh, "RVE_boundary"), (x,t) -> t / last(times) * ɛend * [x[2], 0.0, 0.0], collect(1:dim))
    end


    if bc == :microfree && probtype == :dual
      for i in 1:nslips
            add!(dbcs, Symbol("xi_perp_", i), getnodeset(mesh, "RVE_boundary"), (x,t) -> 0.0)
            if dim == 3
                add!(dbcs, Symbol("xi_o_", i), getnodeset(mesh, "RVE_boundary"), (x,t) -> 0.0)
            end
        end
    end
    close!(dbcs)


    mps = [setup_rand_BCC(l) for i in 1:length(unique(polys))]

    problem = ViscoCrystalPlast.DualProblem(nslips, fev_u, fev_ξ)


    ############################
    # Solve fine scale problem #
    ############################
    pvd_fine = paraview_collection(joinpath(dirname(@__FILE__), "vtks", "grain_mesh_convergence", "shear_DN_$(dim)_$(probtype)_nslips_$(nslips)_N_$(n_polys)_$(bc)"))
    global timestep_fine = 0
    global Ω_grains = zeros(n_polys, length(times)-1)
    #global σs = Matrix{Float64}[]
    #global εs = Matrix{Float64}[]
    #global εps = Matrix{Float64}[]
    #global γs = []

    if probtype == :dual
        if dim == 2
            global msss = ViscoCrystalPlast.CrystPlastDualQD{2,Float64,3}[]
            global homo_ms_grains = Matrix{ViscoCrystalPlast.CrystPlastDualQD{2,Float64,3}}(n_polys, length(times)-1)
        else
            global msss = ViscoCrystalPlast.CrystPlastDualQD{3,Float64,6}[]
            global homo_ms_grains = Matrix{ViscoCrystalPlast.CrystPlastDualQD{3,Float64,6}}(n_polys, length(times)-1)
        end
    else
        if dim == 2
            global msss = ViscoCrystalPlast.CrystPlastPrimalQD{3,Float64,3}[]
        else
            global msss = ViscoCrystalPlast.CrystPlastPrimalQD{3,Float64,6}[]
        end
    end

  exporter_fine = (time, u, mss) ->
      begin
          global timestep_fine += 1
          mss_nodes = ViscoCrystalPlast.move_quadrature_data_to_nodes(mss, mesh, quad_rule)
          # output(problem, pvd_fine, time, timestep_fine, "grain_dir_exp_3D", mesh, dh, u, dbcs, mss_nodes, quad_rule, mps, polys)
          output(problem, pvd_fine, time, timestep_fine, "shear_DN_$(dim)_$(probtype)_nslips_$(nslips)_N_$(n_polys)_$(bc)", mesh, dh, u, dbcs, mss_nodes, quad_rule, mps, polys)

          push!(msss, integrate((element_id, q_point) -> mss[q_point, element_id], mesh, fev_u))
          homo_grains, Ω_grain = integrate_grains((element_id, q_point) -> mss[q_point, element_id], mesh, fev_u, polys)
          homo_ms_grains[:, timestep_fine] = homo_grains
          Ω_grains[:, timestep_fine] = Ω_grain
      end

    sol, mss = ViscoCrystalPlast.solve_problem(problem, mesh, dh, dbcs, fev_u, fev_ξ, mps, times,
                                             exporter_fine, polys)

    vtk_save(pvd_fine)
    save(joinpath(dirname(@__FILE__), "raw_data", "grain_mesh_convergence", "shear_DN_$(dim)_$(probtype)_nslips_$(nslips)_$(n_polys)_$(bc).jld"),
        "mss", reshape(msss, (length(msss), 1)), "homo_ms_grains", homo_ms_grains, "Ω_grains", Ω_grains)
    return # σs,  εs, εps, γs
end
#
