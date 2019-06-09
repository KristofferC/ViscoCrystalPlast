using JuAFEM
using Quaternions
using Tensors
using TimerOutputs
using DataFrames
using JLD

# Check if we are running on a cluster
const RUNNING_SLURM = haskey(ENV, "SLURM_JOBID")
const RUNNING_SLURM_ARRAY = haskey(ENV, "SLURM_ARRAY_TASK_ID")

# Load the "package"
include(joinpath(@__DIR__, "..", "src/ViscoCrystalPlast.jl"))

import ViscoCrystalPlast.Dim

# Function to set material parameters
function setup_rand_BCC()
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

    # Get a random unit vector and an angle
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
 
    l = 0.1
    E = 200000.0
    ν = 0.3
    n = 2.0
    H⟂ = 0.1E
    Ho = 0.1E
    C = 1.0e3
    tstar = 1000.0
    return ViscoCrystalPlast.CrystPlastMP(ViscoCrystalPlast.Dim{3}, E, ν, n, H⟂, Ho, l, tstar, C, slip_directions, slip_planes)
end

# Function to output to vtk file
function output{dim}(time, timestep, filename, mesh, dh, u, dbcs, mss_nodes::AbstractVector, quad_rule::QuadratureRule{dim}, mps, polys)
    nodes_per_ele = dim == 2 ? 3 : 4
    n_sym_components = dim == 2 ? 3 : 6
    tot_nodes = getnnodes(mesh)

    vtkfile = vtk_grid(joinpath(dirname(@__FILE__), "vtks", "$filename" * "_$timestep"), mesh)

    vtk_point_data(vtkfile, dh, u)
    vtk_point_data(vtkfile, dbcs)

    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].σ for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Stress")
    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].ε  for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Strain")
    vtk_point_data(vtkfile, reinterpret(Float64, [mss_nodes[i].ε_p for i in 1:tot_nodes], (n_sym_components, tot_nodes)), "Plastic strain")
    # More things can be exported, will increase file size
    # mp = mps[1]
    #
    #for α in 1:length(mp.s)
    #    vtk_point_data(vtkfile, Float64[mss_nodes[i].γ[α] for i in 1:tot_nodes], "Slip $α")
    #    vtk_point_data(vtkfile, Float64[mss_nodes[i].τ[α] for i in 1:tot_nodes], "Schmid $α")
    #    vtk_point_data(vtkfile, Float64[mss_nodes[i].τ_di[α] for i in 1:tot_nodes], "Tau dissip $α")
    #    vtk_point_data(vtkfile, Float64[mss_nodes[i].χ⟂[α] for i in 1:tot_nodes], "Chi_perp_$α")
    #    vtk_point_data(vtkfile, Float64[mss_nodes[i].χo[α] for i in 1:tot_nodes], "Chi_odot_$α")
    #end
    vtk_cell_data(vtkfile, convert(Vector{Float64}, polys), "poly")
    vtk_save(vtkfile)
end

# Take volume average over a function in a mesh, f should take (element, quadrature_point) as arguments
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

# Compute volume average in each grain
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

# Run the actual simulation
function runit(input_file::String, bctype = ViscoCrystalPlast.Neumann, run_id = 0; do_vtk = true, xi_bc_type = :microhard)
    reset_timer!()
    dim = 3
    # Can fix the seed if we want reproducability
    seed = rand(1:10000)
    srand(seed)
    nslips = 12
    bc = xi_bc_type
    probtype = :dual
    function_space = Lagrange{dim, RefTetrahedron, 1}()
    quad_rule = QuadratureRule{dim, RefTetrahedron}(1)
    fev_u = CellVectorValues(quad_rule, function_space)
    fev_ξ = CellScalarValues(quad_rule, function_space)

    # slip_boundary is created below
    mesh, dh = ViscoCrystalPlast.create_mesh_and_dofhandler(input_file, dim, nslips, probtype)
    polys = ViscoCrystalPlast.create_element_to_grain_map(mesh)

    n_polys = length(unique(polys))
    L_poly = 1.0
    V_poly = L_poly^dim
    V_mesh = V_poly * n_polys
    L_mesh = (V_mesh)^(1/dim)
    new_nodes = similar(getnodes(mesh))
    for n in 1:length(new_nodes)
        new_nodes[n] = JuAFEM.Node(L_mesh * getnodes(mesh, n).x)
    end
    mesh.nodes = new_nodes

    # Some simulation parameters
    Δt = 0.5
    endt = 15.0
    times = 0.0:Δt:endt
    ɛend = 0.3

    dbcs = ViscoCrystalPlast.DirichletBoundaryConditions(dh)
    JuAFEM.addnodeset!(mesh, "RVE_boundary", ViscoCrystalPlast.get_RVE_boundary_nodes(mesh))

    # Simple shear for strain
    ɛ_bar = 0.1 * basevec(Vec{dim}, 1) ⊗ basevec(Vec{dim}, 2)
    ɛ_bar_f(t) = t / last(times) * ɛ_bar
    
    # Boundary conditions for the two different formulations
    if bctype ==  ViscoCrystalPlast.Dirichlet
        add!(dbcs, :u, getnodeset(mesh, "RVE_boundary"), (x,t) -> ɛ_bar_f(t) ⋅ x, collect(1:dim))
    elseif bctype ==  ViscoCrystalPlast.Neumann
        addnodeset!(mesh, "corner", x -> (x ≈ Vec{3}((0.,      0.,      0.))     ||
                                          x ≈ Vec{3}((L_mesh, L_mesh, L_mesh) )));
        add!(dbcs, :u, getnodeset(mesh, "corner"), (x,t) -> ɛ_bar_f(t) ⋅ x, collect(1:dim))
    else
        error("invalid bctype: $bctype")
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


    mps = [setup_rand_BCC() for i in 1:length(unique(polys))]

    # Chose between dual and primal problem here
    problem = ViscoCrystalPlast.DualProblem(nslips, bctype, V_poly, fev_u, fev_ξ)

    timestep_fine = 0
    n_elements = length(dh.grid.cells)

    # Where to store vtk files and exported files
    VTK_PATH = "shear_DN_$(dim)_$(probtype)_$(bctype)_nslips_$(nslips)_ngrains_$(n_polys)_$(bc)"
    if RUNNING_SLURM
        raw_data_path = ENV["SNIC_NOBACKUP"]
    else
        raw_data_path = dirname(@__FILE__)
    end
    JLD_PATH = joinpath(raw_data_path, "raw_data", "shear_DN_$(dim)_$(probtype)_$(bctype)_nslips_$(nslips)_ngrains_$(n_polys)_$(bc)")

    if RUNNING_SLURM_ARRAY
      VTK_PATH *= "_" * string(run_id)
      JLD_PATH *= "_" * string(run_id)
    end

    homo_ms = []
    homo_ms_grains = Matrix(n_polys, length(times)-1)
    dofs = Matrix(JuAFEM.ndofs(dh), length(times)-1)
    all_ms = Array{Any}(getnquadpoints(fev_u), n_elements, length(times)-1)
    solutions = Matrix(n_polys, length(times)-1)
    Ω_grains = Matrix(n_polys, length(times)-1)

    # Create a function to export DataFrames
    exporter_fine = (time, u, mss) -> begin
        @timeit "compute export" begin
            timestep_fine += 1
            # Do integration
            push!(homo_ms, integrate((element_id, q_point) -> mss[q_point, element_id], mesh, fev_u))
            homo_grains, Ω_grain = integrate_grains((element_id, q_point) -> mss[q_point, element_id], mesh, fev_u, polys)
            homo_ms_grains[:, timestep_fine] = homo_grains
            Ω_grains[:, timestep_fine] = Ω_grain
            for i in 1:n_elements
                for qp in 1:getnquadpoints(fev_u)
                    all_ms[qp, i, timestep_fine] = copy(mss[qp, i])
                end
            end
            dofs[:, timestep_fine] = u

            if do_vtk
              mss_nodes = ViscoCrystalPlast.move_quadrature_data_to_nodes(mss, mesh, quad_rule)
              output(time, timestep_fine, VTK_PATH, mesh, dh, u, dbcs, mss_nodes, quad_rule, mps, polys)
            end
        end
    end

    println("Running with n_grains $n_polys, xi_bc: $xi_bc_type, u_bc = $bctype")
    # Solve problem
    conv = true
    try
    sol, σ_bar_n, mss = ViscoCrystalPlast.solve_problem(problem, mesh, dh, dbcs, fev_u, fev_ξ, mps, times,
                                                exporter_fine, polys, ɛ_bar_f)
    catch e
        if !isa(e, ViscoCrystalPlast.IterationException)
            rethrow(e)
        end
        conv = false
        @show e
    end

    if !conv
        JLD_PATH *= "_noconv"
    end

    # Export to JLD
    @timeit "export JLD" begin
        file_path = JLD_PATH
        f = jldopen(string(file_path, ".jld"), "w")
        write(f, "mps", mps)
        #write(f, "dofs", dofs)
        write(f, "seed", seed)
        write(f, "homo_ms", homo_ms)
        #write(f, "all_ms", all_ms)
        write(f, "homo_ms_grains", homo_ms_grains)
        write(f, "Ω_grains", Ω_grains)
        #write(f, "dh", dbcs.dh)
        #write(f, "fev_u", fev_u)
        #write(f, "fev_ξ", fev_ξ)
        #write(f, "grid", dbcs.grid)
        write(f, "input_file", readstring(@__FILE__))
        close(f)
    end

    t = (bctype == ViscoCrystalPlast.Neumann ? "N" : "D")
    t2 = (xi_bc_type == :microhard ? "MH" : "MF")

    print_timer(title = "N = $n_polys id = $run_id bc = $t2 u_bc = $t")
    return
end

runit(joinpath(@__DIR__, "meshes", "n1-id1.inp"))
