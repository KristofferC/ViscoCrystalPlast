import JuAFEM.vtk_grid
import JuAFEM.Grid

const GRID_TYPES = Dict(
    "CPE3" => Triangle,
    "C3D4" => Tetrahedron
)

function Grid(mesh::AbaqusMesh, element_type::String)
    c = mesh.nodes.coordinates
    dim_sum = sum(abs2, c, 2)
    if dim_sum[2] == 0.0
        dim = 1
    elseif dim_sum[3] == 0.0
        dim = 2
    else
        dim = 3
    end

    nodes = reinterpret(Node{dim, Float64}, c[1:dim, :], (size(c, 2),) )

    top =  mesh.elements[element_type].topology
    cells = reinterpret(GRID_TYPES[element_type], top, (size(top, 2),))

    offset = mesh.elements[element_type].numbers[1] - 1

    for (element_set_name, element_set) in mesh.element_sets
        for i in 1:length(element_set)
            element_set[i] -= offset
        end
    end

    new_element_sets = Dict{String, Set{Int}}()
    for (element_set_name, element_set) in mesh.element_sets
        new_element_sets[element_set_name] = Set(element_set)
    end

    new_node_sets = Dict{String, Set{Int}}()
    for (node_set_name, node_set) in mesh.node_sets
        new_node_sets[node_set_name] = Set(node_set)
    end

    Grid(cells, nodes, cellsets = new_element_sets, nodesets = new_node_sets)
end

print_residuals(dh::JuAFEM.DofHandler, f) = print_residuals(STDOUT, dh, f)
function print_residuals(io::IO, dh::JuAFEM.DofHandler, f)
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

    print(io, "|f|: ", @sprintf(": %6.5g ", norm(f)))
    for i in 1:nfields
        print(io, dh.field_names[i], @sprintf(": %6.5g ", residuals[i]))
    end
    print(io, "\n")
end
