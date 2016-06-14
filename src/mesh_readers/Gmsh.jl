const FILE_FORMAT = ("2", "0", "8")

#immutable GeomType
#    gmsh_code :: Int
#    dimen     :: Int
#    nonodes   :: Int
#end
#
#const POINT       = GeomType(15, 1, 1)
#const LINE        = GeomType(1, 1, 2)
#const TRIANGLE    = GeomType(2, 2, 3)
#const TETRAHEDRON = GeomType(4, 3, 4)
#
#const GETGEOMTYPE = Dict(1 => LINE,
#                         2 => TRIANGLE,
#                         4 => TETRAHEDRON,
#                         15 => POINT)
##

immutable GmshNode
    number::Int
    x::Float64
    y::Float64
    z::Float64
end

immutable GmshElement
    number::Int
    eletype::Int
    ntags::Int
    tags::Vector{Int}
    nodes::Vector{Int}
end

immutable RawGmshMesh
    nodes::Vector{GmshNode}
    elements::Vector{GmshElement}
end

"""
mesh = read_msh_file("filename.msh")

Read a .msh file created by Gmsh and return the corresponding
Mesh object.
"""
function read_msh_file(fname::ASCIIString)
    f = open(fname,"r")
    nodes = Vector{GmshNode}()
    elements = Vector{GmshElement}()
    while !eof(f)
        header = eat_line(f)
        if header == "\$MeshFormat"
            fmt = split(eat_line(f))
            for j = 1:3
                @assert fmt[j] == FILE_FORMAT[j]
            end

            if peek_line(f) ==  "\$Comments"
                while eat_line(f) != "\$EndComments"
                    if eof(f)
                        error("found end of file while reading comments")
                    end
                end
            end
            eat_line(f, "\$EndMeshFormat")

        elseif header == "\$Nodes"
            nnodes = parse(Int, eat_line(f))
            resize!(nodes, nnodes)
            for i in 1:nnodes
                n, x, y, z = parse_line(eat_line(f), Int, Float64, Float64, Float64)
                nodes[i] = GmshNode(n, x, y, z)
            end
            eat_line(f, "\$EndNodes")

        elseif header == "\$Elements"
            nelements = parse(Int, eat_line(f))
            resize!(elements, nelements)
            for i in 1:nelements
                ele_data = map(x -> parse(Int, x), split(eat_line(f)))
                n_tags = ele_data[3]
                n, ele_type, n_tags, tags, topology = ele_data[1], ele_data[2], ele_data[3], ele_data[4:4+n_tags-1], ele_data[4+n_tags:end]
                @assert length(topology) > 0
                elements[i] = GmshElement(n, ele_type, n_tags, tags, topology)
            end
            eat_line(f, "\$EndElements")
        else
            error("Unknown header: $header")
        end
    end
    return RawGmshMesh(nodes, elements)
end

