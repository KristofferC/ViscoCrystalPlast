# Functionality for reading a mesh from COMSOL
# Currently only 1 object, no geometric entity information
module ComsolMeshReader

type ComsolMesh{dim}
    version::UTF8String
    space_dim::Int
    coordinates::Vector{NTuple{dim, Float64}}
    elements::Dict{UTF8String, Vector}
end

type LineFeeder
    file::IOStream
    next_line::UTF8String
end

LineFeeder(f::IOStream) = LineFeeder(f, readline(f))

function has_next_line(lf::LineFeeder)
    if !isempty(lf.next_line)
        return true
    else
        return false
    end
end

function get_next_line(lf::LineFeeder)
    if lf.next_line != ""
        nl = lf.next_line
        lf.next_line = readline(lf.file)
        while lf.next_line == "\n"
            if isempty(lf.next_line)
                @goto err
            end
            lf.next_line = readline(lf.file)
        end
        return strip(nl)
    else
        @label err
        throw(error("Unexpected end"))
    end
end


function err(s)
    throw(error("Unexpcted line read: $s"))
end

function read_mphtxt(filename, verbosity::Int=0)
    line_feed = LineFeeder(open(filename, "r"))

    next_line = get_next_line(line_feed)
    if !startswith(next_line, "# Created by COMSOL")
        err(next_line)
    end

    next_line = get_next_line(line_feed)
    if !startswith(next_line, "# Major & minor version")
        err(next_line)
    end
    version = get_next_line(line_feed)

    next_line = get_next_line(line_feed)
    while !endswith(next_line, "# sdim")
        next_line = get_next_line(line_feed)
    end
    sdim = parse(Int, split(next_line)[1])

    next_line = get_next_line(line_feed)
    if !endswith(next_line, "# number of mesh points")
        err(next_line)
    end
    n_mesh_points = parse(Int, split(next_line)[1])

    get_next_line(line_feed) # Skip #lowest mesh point

    next_line = get_next_line(line_feed)
    if !startswith(next_line, "# Mesh point coordinates")
        err(next_line)
    end

    ####################
    # Read coordinates #
    ####################
    coords = Vector{NTuple{sdim, Float64}}(n_mesh_points)
    for i in 1:n_mesh_points
        next_line = get_next_line(line_feed)
        coords[i] = (map(i->parse(Float64, i), split(next_line))...)
    end

    next_line = get_next_line(line_feed)
    if !endswith(next_line, "# number of element types")
        err(next_line)
    end
    n_element_types = parse(Int, split(next_line)[1])

    elements = Dict{UTF8String, Vector}()

    ######################
    # Read element types #
    ######################
    for i in 1:n_element_types
        get_next_line(line_feed) # Ignore # Type #x

        next_line = get_next_line(line_feed)
        if !endswith(next_line, "# type name")
            err(next_line)
        end
        type_name = split(next_line, " #")[1]

        next_line = get_next_line(line_feed)
        if !endswith(next_line, "# number of nodes per element")
            err(next_line)
        end
        n_node_per_element = parse(Int, split(next_line, " #")[1])

        next_line = get_next_line(line_feed)
        if !endswith(next_line, "# number of elements")
            err(next_line)
        end
        n_elements = parse(Int, split(next_line, " #")[1])
        element_vertices = Vector{NTuple{n_node_per_element, Int}}(n_elements)

        next_line = get_next_line(line_feed)
        if !endswith(next_line, "# Elements")
            err(next_line)
        end

        #################
        # Read elements #
        #################
        for i in 1:n_elements
            next_line = get_next_line(line_feed)
            element_vertices[i] = (map(i->(parse(Int, i)+1), split(next_line))...)
        end

        get_next_line(line_feed) # Skip geometric entity
        elements[type_name] = element_vertices
    end

    return ComsolMesh(version, sdim, coords, elements)
end

end