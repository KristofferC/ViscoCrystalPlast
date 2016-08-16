
import MeshIO.AbaqusMesh
import JuAFEM.vtk_grid

immutable GeometryMesh{dim}
    coords::Vector{Vec{dim, Float64}}
    topology::Matrix{Int}
    element_sets::Dict{String, Vector{Int}}
    node_sets::Dict{String, Vector{Int}}
end

nnodes(m::GeometryMesh) = length(m.coords)
nelements(m::GeometryMesh) = size(m.topology, 2)

element_set(m::GeometryMesh, s::String) = m.element_sets[s]
add_element_set!(m::GeometryMesh, s::String, v::Vector{Int}) = m.element_sets[s] = v
add_node_set!(m::GeometryMesh, s::String, v::Vector{Int}) = m.node_sets[s] = v
node_set(m::GeometryMesh, s::String) = m.node_sets[s]

function GeometryMesh(mesh::AbaqusMesh, element_type::String)
    c = mesh.nodes.coordinates
    dim_sum = sumabs2(c, 2)
    if dim_sum[2] == 0.0
        dim = 1
    elseif dim_sum[3] == 0.0
        dim = 2
    else
        dim = 3
    end

    nnodes = size(c, 2)
    coords = reinterpret(Vec{dim, Float64}, c[1:dim, :], (nnodes,) )

    offset = mesh.elements[element_type].numbers[1] - 1

    for (element_set_name, element_set) in mesh.element_sets
        for i in 1:length(element_set)
            element_set[i] -= offset
        end
    end

    GeometryMesh(copy(coords), copy(mesh.elements[element_type].topology),
                 mesh.element_sets, mesh.node_sets)
end


function vtk_grid{dim}(mesh::GeometryMesh{dim}, filename; compress = true, append = true)
    c_mat = reinterpret(Float64, mesh.coords, (dim, length(mesh.coords)))
    JuAFEM.vtk_grid(mesh.topology, c_mat, filename, compress = compress, append = append)
end

function element_coordinates{dim}(gm::GeometryMesh{dim}, ele::Int)
    n_nodes = size(gm.topology, 1)
    c = [zero(Vec{dim, Float64}) for i in 1:n_nodes]
    element_coordinates!(c, gm, ele)
end

function element_coordinates!{dim}(coords::Vector{Vec{dim, Float64}}, gm::GeometryMesh, ele::Int)
    n_nodes = size(gm.topology, 1)
    @assert length(coords) == n_nodes
    for i in 1:n_nodes
        coords[i] = gm.coords[gm.topology[i, ele]]
    end
    return coords
end

function node_coordinates(gm::GeometryMesh, node::Int)
    return gm.coords[node]
end

function element_vertices(gm::GeometryMesh, element::Int)
    return gm.topology[:, element]
end



#function node_neighboring_elements(mesh)
#    neighboring_elements = [Set{Int}() for i in 1:size(mesh.coords, 2)]
#
#    for e in 1:size(mesh.topology, 2)
#        for node in mesh.topology[:, e]
#            push!(neighboring_elements[node], e)
#        end
#    end
#    return neighboring_elements
#end

type DofHandler{dim}
    dofs_nodes::Matrix{Int}
    dofs_elements::Matrix{Int}
    field_names::Vector{Symbol}
    dof_dims::Vector{Int}
    closed::Bool
    dofs_vec::Vector{Int}
    mesh::GeometryMesh{dim}
end

function DofHandler(m::GeometryMesh)
    DofHandler(Matrix{Int}(), Matrix{Int}(), Symbol[], Int[], false, Int[], m)
end
ndofs(dh::DofHandler) = length(dh.dofs_nodes)
isclosed(dh::DofHandler) = dh.closed
dofs_node(dh::DofHandler, i::Int) = dh.dof_nodes[:, i]


function dofs_element(dh::DofHandler, i::Int)
    @dbg_assert isclosed(dh)
    return dh.dofs_elements[:, i]
end



function add_field!(dh::DofHandler, names::Vector{Symbol}, dims)
    @assert length(names) == length(dims)
    for i in 1:length(names)
        add_field!(dh, names[i], dims[i])
    end
end

function add_field!(dh::DofHandler, name::Symbol, dim::Int)
    @dbg_assert !isclosed(dh)
    if name in dh.field_names
        error("duplicate field name")
    end

    push!(dh.field_names, name)
    push!(dh.dof_dims, dim)
    append!(dh.dofs_vec, length(dh.dofs_vec)+1:length(dh.dofs_vec) +  dim * nnodes(dh.mesh))

    return dh
end

function dof_offset(dh::DofHandler, field_name::Symbol)
    offset = 0
    i = 0
    for name in dh.field_names
        i += 1
        if name == field_name
            return offset
        else
            offset += dh.dof_dims[i]
        end
    end
    error("unexisting field name $field_name among $(dh.field_names)")
end

function ndim(dh::DofHandler, field_name::Symbol)
    i = 0
    for name in dh.field_names
        i += 1
        if name == field_name
            return dh.dof_dims[i]
        end
    end
    error("unexisting field name $field_name among $(dh.field_names)")
end

function close!(dh::DofHandler)
    @assert !isclosed(dh)
    dh.dofs_nodes = reshape(dh.dofs_vec, (length(dh.dofs_vec) ÷ nnodes(dh.mesh), nnodes(dh.mesh)))
    add_element_dofs!(dh)
    dh.closed = true
    return dh
end

function add_element_dofs!(dh::DofHandler)
    n_elements = size(dh.mesh.topology, 2)
    n_vertices = size(dh.mesh.topology, 1)
    element_dofs = Int[]
    ndofs = size(dh.dofs_nodes, 1)
    for element in 1:n_elements
        offset = 0
        for dim_doftype in dh.dof_dims
            for node in view(dh.mesh.topology, :, element)
                for j in 1:dim_doftype
                    push!(element_dofs, dh.dofs_nodes[offset + j, node])
                end
            end
            offset += dim_doftype
        end
    end
    dh.dofs_elements = reshape(element_dofs, (ndofs * n_vertices, n_elements))
end

function vtk_point_data(vtkfile, dh, u)
    offset = 0
    for i in 1:length(dh.field_names)
        ndim_field = dh.dof_dims[i]
        space_dim = ndim_field == 2 ? 3 : ndim_field
        data = zeros(space_dim, nnodes(dh.mesh))
        for j in 1:size(dh.dofs_nodes, 2)
            for k in 1:ndim_field
                data[k, j] = u[dh.dofs_nodes[k + offset, j]]
            end
        end
        vtk_point_data(vtkfile, data, string(dh.field_names[i]))
        offset += ndim_field
    end
    return vtkfile
end

function print_residuals(dh, f)
    nfields = length(dh.field_names)
    residuals = zeros(nfields)
    offset = 0
    for i in 1:nfields
        ndim_field = dh.dof_dims[i]
        for j in 1:size(dh.dofs_nodes, 2)
            for k in 1:ndim_field
                residuals[i] += f[dh.dofs_nodes[k + offset, j]]^2
            end
        end
        offset += ndim_field
        residuals[i] = sqrt(residuals[i])
    end

    print("|f|: ", @sprintf(": %6.5g ", norm(f)))
    for i in 1:nfields
        print(dh.field_names[i], @sprintf(": %6.5g ", residuals[i]))
    end
    print("\n")
end
