using DataStructures

function create_mesh_and_dofhandler(inputfile, dim, nslips, probtype)
    m = load(inputfile)
    println(size(m.nodes.coordinates))
    m, old_to_new = ViscoCrystalPlast.update_mesh!(m, dim)
    println(size(m.nodes.coordinates))
    condensed_mesh, duplicated_nodes = ViscoCrystalPlast.condense_mesh(m, old_to_new, dim)
    bulk_element_name = dim == 2 ? "CPE3" : "C3D4"
    geomesh = ViscoCrystalPlast.GeometryMesh(condensed_mesh, bulk_element_name)
    dh = ViscoCrystalPlast.add_grad_dofs!(geomesh, duplicated_nodes, nslips, probtype)
    return geomesh, dh
end

function get_RVE_boundary_nodes(mesh::ViscoCrystalPlast.GeometryMesh)
    nodes_on_RVE_edge = Int[]

    for (name, nodes) in mesh.node_sets
        if !(name in ("x1", "x0", "y1", "y0", "z1",  "z0"))
            continue
        end
        append!(nodes_on_RVE_edge, nodes)
    end
    return unique(nodes_on_RVE_edge)
end

# Internal nodes that are on the edge between two grains
function get_grain_boundary_nodes{dim}(mesh::ViscoCrystalPlast.GeometryMesh{dim})
    grain_name = dim == 2 ? "face" : "poly"
    nodes_grain = [Int[] for i in 1:length(mesh.element_sets)]

    curr_set = 0
    for (name, elements) in mesh.element_sets
        !startswith(name, grain_name) && continue
        p = parse(Int, name[5:end])
        for element in elements
            for v in ViscoCrystalPlast.element_vertices(mesh, element)
                push!(nodes_grain[p], v)
            end
        end
    end

    nodes_grain = [unique(x) for x in nodes_grain]
    println(nodes_grain)

    n_grains = zeros(Int, ViscoCrystalPlast.nnodes(mesh))
    for (i, nodes_in_grain) in enumerate(nodes_grain)
        for node in nodes_in_grain
            n_grains[node] += 1
        end
    end

    nodes_in_boundary = find(x -> x > 1, n_grains)
    return nodes_in_boundary
end

# Creates poly such that poly[element_id] = p where p is the grain
function create_element_to_grain_map{dim}(mesh::ViscoCrystalPlast.GeometryMesh{dim})
    grain_name = dim == 2 ? "face" : "poly"
    poly = zeros(Int, ViscoCrystalPlast.nelements(mesh))
    for (name, elements) in mesh.element_sets
        if !startswith(name, grain_name)
            continue
        else
            p = parse(Int, name[5:end])
            for element in elements
                poly[element] = p
            end
        end
    end
    return poly
end




function get_nodes_in_all_face_sets(mesh, dim)
    """
    This function finds all nodes that sits in a face.
    :param mesh:
    :type mesh: :class:`Mesh`
    :return: The node identifiers in all faces
    :rtype: list[ints]
    """
    nodes_in_face_sets = Set{Int}()
    grain_name = dim == 2 ? "face" : "poly"
    for (element_set_name, element_set) in mesh.element_sets
        !startswith(element_set_name, grain_name) && continue
        for node in element_set
            push!(nodes_in_face_sets, node)
        end
    end
    return collect(nodes_in_face_sets)
end

function get_node_id_grain_lut(mesh, dim)
    """
    This function creates a (default) dictionary that
    works as a lookup table for what grains contain
    what nodes.
    """
    grain_name = dim == 2 ? "face" : "poly"
    element_name = dim == 2 ? "CPE3" : "C3D4"
    bulk_elements = mesh.elements[element_name]
    offset = bulk_elements.numbers[1] - 1 # Assuming numbers are consequtive 4,5,6,7...

    d = OrderedDict{Int, Vector{Int}}()
    for i in 1:length(mesh.nodes.numbers)
        d[i] = Vector{Int}()
    end
    for (element_set_name, element_set) in mesh.element_sets
        !startswith(element_set_name, grain_name) && continue
        grain = parse(Int, element_set_name[5:end])
        for element_id in element_set
            vertices = bulk_elements.topology[:, element_id - offset]
            for node_id in vertices
                if !(grain in d[node_id])
                    push!(d[node_id], grain)
                end
            end
        end
    end
    return d
end


function get_grains_connected_to_face(mesh,
                                      face_set::Vector{Int}, node_id_grain_lut, dim::Int)
    """
    This function find the grain connected to the face set given as argument.
    Three nodes on a grain boundary can all be intersected by one grain
    in which case the grain face is on the boundary or by two grains. It
    is therefore sufficient to look at the set of grains contained by any
    three nodes in the face set and take the intersection of these sets.
    """

    element_name = dim == 2 ? "T3D2" : "CPE3"
    grains_connected_to_face = Set{Int}[]
    low_dim_elements = mesh.elements[element_name]
    offset = low_dim_elements.numbers[1] - 1 # Assuming numbers are consequtive 4,5,6,7...

    for node_id in low_dim_elements.topology[:, face_set[1] - offset]
        grains_with_node_id = node_id_grain_lut[node_id]
        push!(grains_connected_to_face, Set(grains_with_node_id))
    end
    return collect(intersect(grains_connected_to_face...))
end


function get_ele_and_grain_with_node_id(
        mesh, node_id, grain_id_1, grain_id_2, dim::Int):
    """
    Find the elements that has vertices with the node identifier node_id
    and it belongs to grain with identifier grain_id_1 or grain_id_2
    """

    bulk = Int[]
    grains = Int[]
    set_type = dim == 2 ? "face" : "poly"
    element_name = dim == 2 ? "CPE3" : "C3D4"
    bulk_elements = mesh.elements[element_name]
    offset = bulk_elements.numbers[1] - 1

    for grain_id in (grain_id_1, grain_id_2)
        for element_id in mesh.element_sets[set_type * string(grain_id)]
            if node_id in bulk_elements.topology[:, element_id - offset]
                push!(bulk, element_id)
                push!(grains, grain_id)
            end
        end
    end
    return grains, bulk
end

function nodes_in_faceset(mesh, ele_name::String, faceset::Vector{Int})
    bulk_elements = mesh.elements[ele_name]
    offset = bulk_elements.numbers[1] - 1 # Assuming numbers are consequtive 4,5,6,7...
    nodes = Int[]
    for element in faceset
        for t in bulk_elements.topology[:, element - offset]
            push!(nodes, t)
        end
    end
    return unique(nodes)
end



# Inserts extra nodes in the middle between grains
function update_mesh!(mesh, dim)
    """
    Creates and inserts cohesive elements between the grains in the mesh.
    The element sets, ordering of vertices in elements etc etc need to
    follow the convention from Neper.
    """

    set_type = dim == 2 ? "edge" : "face"
    bulk_element_name = dim == 2 ? "CPE3" : "C3D4"
    face_element_name = dim == 2 ? "T3D2" : "CPE3"

    bulk_elements = mesh.elements[bulk_element_name]
    offset = bulk_elements.numbers[1] - 1 # Assuming numbers are consequtive 4,5,6,7...
    node_id_grain_lut = get_node_id_grain_lut(mesh, dim)
    new_node_dict = OrderedDict{Tuple{Int, Int}, Int}()
    new_to_old = OrderedDict{Int, Int}()
    old_to_new = OrderedDict{Int, Vector{Int}}()
    n_nodes = length(mesh.nodes.numbers)
    slip_boundary = Int[]

    new_node_numbers = Int[]
    did_something = false
    for (face_set_name, face_set) in mesh.element_sets
        !startswith(face_set_name, set_type) && continue
        did_something = true

        grains_connected_to_face = get_grains_connected_to_face(mesh,
                                                                face_set,
                                                                node_id_grain_lut,
                                                                dim)

        # Ignore sets at boundary
        length(grains_connected_to_face) == 1 && continue

        @assert length(grains_connected_to_face) == 2
        grain_id_1, grain_id_2 = grains_connected_to_face

        # For each node in face make two new at the same place

        for node_id in nodes_in_faceset(mesh, face_element_name, face_set)
            #node_coords = mesh.nodes.coordinates[:, node_id]
            for grain_id in (grain_id_1, grain_id_2)
                if !haskey(new_node_dict, (grain_id, node_id))
                    n_nodes += 1
                    new_node_dict[grain_id, node_id] = n_nodes
                    new_to_old[n_nodes] = node_id
                    push!(slip_boundary, n_nodes)
                    if !haskey(old_to_new, node_id)
                        old_to_new[node_id] = Vector{Int}()
                    end
                    push!(old_to_new[node_id], n_nodes)
                end
            end


            # Reconnect the bulk element with vertices in the node that is being duplicated
            # to one of the new nodes.
            grain_ids, element_ids = get_ele_and_grain_with_node_id(
                mesh, node_id, grain_id_1, grain_id_2, dim)

            for (grain_id, element_id) in zip(grain_ids, element_ids)
                idx = findfirst(bulk_elements.topology[:, element_id - offset], node_id)
                bulk_elements.topology[idx, element_id - offset] = new_node_dict[grain_id, node_id]
            end
        end
    end
    if !did_something
        error("We didnt do anything, check your input file")
    end
    # Need to add the new nodes to the mesh
    new_coords = Float64[]
    node_numbers = Int[]
    for (new_node, old_node) in new_to_old
        append!(new_coords, mesh.nodes.coordinates[:, old_node])
        push!(node_numbers, new_node)
    end

    append!(mesh.nodes.numbers, node_numbers)
    mesh.nodes.coordinates = hcat(mesh.nodes.coordinates, reshape(new_coords, (3, length(new_coords) ÷ 3)))
    mesh.node_sets["slip_boundary"] = slip_boundary
    return mesh, old_to_new
end

# Add dofs for the specific grad problem
function add_grad_dofs!{dim}(mesh::ViscoCrystalPlast.GeometryMesh{dim}, duplicated_nodes::Vector{Vector{Int}}, nslips, probtype)
    @assert probtype in (:dual, :primal)

    dh = ViscoCrystalPlast.DofHandler(mesh)
    ViscoCrystalPlast.add_field!(dh, :u, dim)
    if probtype == :primal
        ViscoCrystalPlast.add_field!(dh, [Symbol("slip_", i) for i in 1:nslips], (ones(Int, nslips)...))
    else
        ViscoCrystalPlast.add_field!(dh, [Symbol("xi_perp_", i) for i in 1:nslips], (ones(Int, nslips)...))
        if dim == 3
            ViscoCrystalPlast.add_field!(dh, [Symbol("xi_o_", i) for i in 1:nslips], (ones(Int, nslips)...))
        end
    end

    all_nodes = 1:length(mesh.coords)
    nodes_without_slip = setdiff(all_nodes, mesh.node_sets["slip_boundary"])

    grad_dofs = probtype == :dual ? dim - 1 : 1
    dofs_per_node = dim + grad_dofs * nslips

    dofs_nodes = Matrix{Int}(dofs_per_node, length(all_nodes))
    fill!(dofs_nodes, -1)
    dof_number = 1
    for n in nodes_without_slip
        for dof in 1:dofs_per_node
            dofs_nodes[dof, n] = dof_number
            dof_number += 1
        end
    end
    for dup_nodes in duplicated_nodes
        for i in 1:dim
            for node in dup_nodes
                @dbg_assert dofs_nodes[i, node] == -1
                dofs_nodes[i, node] = dof_number
            end
            dof_number += 1
        end

        for node in dup_nodes
            for i in 1:grad_dofs * nslips
                @dbg_assert dofs_nodes[dim + i, node] == -1
                dofs_nodes[dim + i, node] = dof_number
                dof_number += 1
            end
        end
    end

    dh.dofs_nodes = dofs_nodes
    @dbg_assert findfirst(dofs_nodes, -1) == 0

    add_element_dofs!(dh)
    dh.closed = true
    return dh
end

# Condenses the node numbers such that they start from 1 again
# Returns a new AbaqusMesh
function condense_mesh(mesh, old_to_new, dim)
    node_nr = 1
    element_nr = 1
    bulk_element_name = dim == 2 ? "CPE3" : "C3D4"

    bulk_elements = mesh.elements[bulk_element_name]
    new_topology = copy(bulk_elements.topology)
    new_coordinates = Float64[]
    new_node_numbers = Int[]
    new_element_numbers = Int[]
    renumber = Dict{Int, Int}()
    renumber_elements = Dict{Int, Int}()
    for element in 1:size(bulk_elements.topology, 2)
        for j in 1:size(bulk_elements.topology, 1)
            vertex = bulk_elements.topology[j, element]
            if !haskey(renumber, vertex)
                renumber[vertex] = node_nr
                node_nr += 1
                push!(new_node_numbers, node_nr)
                append!(new_coordinates, mesh.nodes.coordinates[:, vertex])
            end
            new_topology[j, element] = renumber[vertex]
        end
        push!(new_element_numbers, element_nr)
        renumber_elements[bulk_elements.numbers[element]] = element_nr
        element_nr += 1
    end


    duplicated_nodes = Vector{Int}[]
    for (old, news) in old_to_new
        v = Int[]
        for n in news
            push!(v, renumber[n])
        end
        push!(duplicated_nodes, v)
    end

    new_node_sets = Dict{String, Vector{Int}}()
    for (node_set_name, node_set) in mesh.node_sets
        node_set_nodes = Int[]
        new_node_sets[node_set_name] = node_set_nodes
        for node in node_set
            if haskey(old_to_new, node)
                node_vec = old_to_new[node]
                for new_node in node_vec
                    push!(node_set_nodes, renumber[new_node])
                end
            else
                push!(node_set_nodes, renumber[node])
            end
        end
    end

    set_type = dim == 2 ? "face" : "poly"
    new_element_sets = Dict{String, Vector{Int}}()
    for (element_set_name, element_set) in mesh.element_sets
        !startswith(element_set_name, set_type) && continue
        element_set_verts = Int[]
        new_element_sets[element_set_name] = element_set_verts
        for element in element_set
            push!(element_set_verts, renumber_elements[element])
        end
    end

    new_nodes = MeshIO.AbaqusNodes(new_node_numbers, reshape(new_coordinates, (3, length(new_coordinates) ÷ 3)))
    new_elements = Dict(bulk_element_name => MeshIO.AbaqusElements(new_element_numbers, new_topology))

    return MeshIO.AbaqusMesh(new_nodes, new_elements, new_node_sets, new_element_sets), duplicated_nodes
end
