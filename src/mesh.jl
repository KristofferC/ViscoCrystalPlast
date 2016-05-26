immutable GeometryMesh
    coords::Matrix{Float64}
    topology::Matrix{Int}
    boundary_nodes::Vector{Int}
end

function element_coordinates(gm::GeometryMesh, ele::Int)
    dim = size(gm.coords, 1)
    n_nodes = size(gm.topology, 1)
    element_coordinates!(zeros(dim, n_nodes), gm, ele)
end

function element_coordinates!(coords::Matrix{Float64}, gm::GeometryMesh, ele::Int)
    dim = size(gm.coords, 1)
    n_nodes = size(gm.topology, 1)
    @assert size(coords) == (dim, n_nodes)
    for i in 1:n_nodes
        coords[:, i] = gm.coords[:, gm.topology[i, ele]]
    end
    return coords
end


function create_mesh(mesh_file::AbstractString)
    mesh = ComsolMeshReader.read_mphtxt(mesh_file)
    modify_mesh(mesh)
end

function modify_mesh(mesh)
    dim = mesh.space_dim
    coordinates = mesh.coordinates
    n_nodes = length(coordinates)
    if dim == 2
        tri_elements = mesh.elements["3 tri"]
        boundary_elements = mesh.elements["3 edg"]
        nnodes = 3
        topology = reinterpret(Int, tri_elements, (nnodes, length(tri_elements)))
    elseif dim == 3
        tri_elements = mesh.elements["3 tet"]
        boundary_elements = mesh.elements["3 tri"]
        nnodes = 4
        topology = reinterpret(Int, tri_elements, (nnodes, length(tri_elements)))
    end

    boundary_nodes = unique(reinterpret(Int, boundary_elements, (dim*length(boundary_elements),)))
    print(typeof(coordinates))
    coords_mat = reinterpret(Float64, coordinates, (dim, length(coordinates)))

    mesh = GeometryMesh(coords_mat, topology, boundary_nodes)

   return mesh
end

immutable Dofs
    dof_ids::Matrix{Int}
    dof_types::Vector{Symbol}
end


function dofs_node(gm::GeometryMesh, dofs::Dofs, i::Int)
    return dofs.dof_ids[:, i]
end

function dofs_element(gm::GeometryMesh, dofs::Dofs, i::Int)
    element_dofs = Int[]
    for node in gm.topology[:, i]
        append!(element_dofs, dofs_node(gm, dofs, node))
    end
    return element_dofs
end

function add_dofs(mesh::GeometryMesh, fields::Vector{Symbol})
    dofs_per_node = length(fields)
    n_nodes = size(mesh.coords,2)
    dofs = reshape(collect(1:dofs_per_node*n_nodes), (dofs_per_node, n_nodes))
    return Dofs(dofs, fields)
end

function node_neighboring_elements(mesh)
    neighboring_elements = [Set{Int}() for i in 1:size(mesh.coords, 2)]

    for e in 1:size(mesh.topology, 2)
        for node in mesh.topology[:, e]
            push!(neighboring_elements[node], e)
        end
    end
    return neighboring_elements
end