
immutable AbaqusElement
    n::Int
    topology::Vector{Int}
end

immutable AbaqusNode
    n::Int
    x::Float64
    y::Float64
    z::Float64
end

immutable AbaqusMesh
    nodes::Vector{AbaqusNode}
    elements::Dict{String, Vector{AbaqusElement}}
    node_sets::Dict{String, Vector{Int}}
    element_sets::Dict{String, Vector{Int}}
end

function to_viscomesh(am::AbaqusMesh)
    eles = am.elements["C3D4"]

    coords = zeros(3, length(am.nodes))
    for i in 1:length(am.nodes)
        node = am.nodes[i]
        coords[:, i] = [node.x, node.y, node.z]
    end

    topology = zeros(Int, 4, length(eles))
    for i in 1:length(eles)
        topology[:, i] = eles[i].topology
    end



    return GeometryMesh(coords, topology, boundary_nodes), polytype

end

iskeyword(l) = startswith(l, "*")

"""
mesh = read_msh_file("filename.msh")

Read a .msh file created by Gmsh and return the corresponding
Mesh object.
"""
function read_inp_file(fname::ASCIIString)
    f = open(fname,"r")
    nodes = Vector{AbaqusNode}()
    elements = Dict{String, Vector{AbaqusElement}}()
    node_sets = Dict{String, Vector{Int}}()
    element_sets = Dict{String, Vector{Int}}()
    while !eof(f)
        header = eat_line(f)
        if header == ""
            continue
        end
        if ((m = match(r"\*Part, name=(.*)", header)) != nothing)

        elseif ((m = match(r"\*Node", header)) != nothing)
            while !iskeyword(peek_line(f))
                l = strip(eat_line(f))
                l == "" && continue
                n, x, y, z = parse_line(l, Int, Float64, Float64, Float64)
                push!(nodes, AbaqusNode(n, x, y, z))
            end

        elseif ((m = match(r"\*Element, type=(.*)", header)) != nothing)
            type_elements = Vector{AbaqusElement}()
            while !iskeyword(peek_line(f))
                l = strip(eat_line(f))
                l == "" && continue
                l_split = split(l, [','])
                ele_data = map(x -> parse(Int, x), l_split)
                push!(type_elements, AbaqusElement(ele_data[1], ele_data[2:end]))
            end
            elements[m.captures[1]] = type_elements

        elseif ((m = match(r"\*Elset, elset=(.*)", header)) != nothing)
            buf = IOBuffer()
            while !iskeyword(peek_line(f))
                print(buf, eat_line(f))
            end
            buf_str = strip(takebuf_string(buf))
            element_sets[m.captures[1]] = map(x -> parse(Int, x), split(buf_str, [',']))

        elseif ((m = match(r"\*Nset, nset=(.*)", header)) != nothing)
            buf = IOBuffer()
            while !iskeyword(peek_line(f))
                print(buf, eat_line(f))
            end
            buf_str = strip(takebuf_string(buf))
            node_sets[m.captures[1]] = map(x -> parse(Int, x), split(buf_str, [',']))

        elseif ((m = match(r"\*End Part", header)) != nothing)
            l = eat_line(f)
        else
            error("Unknown header: $header")
        end
    end
    return AbaqusMesh(nodes, elements, node_sets, element_sets)
end



