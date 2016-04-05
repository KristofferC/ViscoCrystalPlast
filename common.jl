using ContMechTensors
using Parameters
using JuAFEM

if !isdefined(:CrystPlastMP)
    @eval begin
        type CrystPlastMP{dim, T, M, N}
            E::T
            Ee::SymmetricTensor{4, dim, T, N}
            ν::T
            n::T
            H⟂::T
            Ho::T
            Hgrad::Vector{SymmetricTensor{2, dim, T, M}}
            lα::T
            tstar::T
            C::T
            angles::Vector{T}
            sxm_sym::Vector{SymmetricTensor{2, dim, T, M}}
            s::Vector{Vec{dim, T}}
            m::Vector{Vec{dim, T}}
            l::Vector{Vec{dim, T}}
        end
    end
end



function CrystPlastMP{T, dim}(::Type{Dim{dim}}, E::T, ν::T, n::T, H⟂::T, Ho::T, lα::T, tstar::T, C::T, angles::Vector{T})
    nslip = length(angles)
    sxm_sym = SymmetricTensor{2, dim}[]
    s = Vector{Vec{dim, T}}(nslip)
    m = Vector{Vec{dim, T}}(nslip)
    l = Vector{Vec{dim, T}}(nslip)
    for α = 1:nslip
        t = deg2rad(angles[α])
        s_a = Tensor{1, 3}([cos(t), sin(t), 0.0])
        m_a = Tensor{1, 3}([cos(t + pi/2), sin(t + pi/2), 0.0])
        s[α] = convert(Tensor{1, dim}, s_a)
        m[α] = convert(Tensor{1, dim}, m_a)
        l[α] = convert(Tensor{1, dim}, Tensor{1, 3}(cross(s_a, m_a)))
    end

    sxm_sym = [convert(SymmetricTensor{2, dim}, s[α] ⊗ m[α]) for α in 1:nslip]
    Hgrad = [convert(SymmetricTensor{2, dim}, H⟂ * s[α] ⊗ s[α] + Ho * l[α] ⊗ l[α]) for α in 1:nslip]
    λ = E*ν / ((1+ν) * (1 - 2ν))::Float64
    μ = E / (2(1+ν))::Float64
    δ(i,j) = i == j ? 1.0 : 0.0
    f(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l) * δ(j,k))
    Ee = SymmetricTensor{4, dim}(f)
    CrystPlastMP(E, Ee, ν, n, H⟂, Ho, Hgrad, lα, tstar, C, angles, sxm_sym, s, m, l)
end

function setup_material{dim}(::Type{Dim{dim}})
    E = 200000.0
    ν = 0.3
    n = 2.0
    lα = 0.5
    H⟂ = 0.1E
    Ho = 0.1E
    C = 1.0e3
    tstar = 1000.0
    angles = [20.0, 40.0]
    mp = CrystPlastMP(Dim{dim}, E, ν, n, H⟂, Ho, lα, tstar, C, angles)
    return mp
end


function u_dofs(dofs_u, nnodes, dofs_g, nslip)
    dofs = Int[]
    count = 0
    for i in 1:nnodes
        for j in 1:dofs_u
            push!(dofs, count+j)
        end
        count += nslip * dofs_g + dofs_u
    end
    return dofs
end

function g_dofs(dofs_u, nnodes, dofs_g, nslip, slip)
    dofs = Int[]
    count = dofs_u + (slip -1) * dofs_g
    for i in 1:nnodes
        for j in 1:dofs_g
            push!(dofs, count+j)
        end
        count += dofs_u + nslip * dofs_g
    end
    return dofs
end

abstract AbstractCALMesh


if !isdefined(:CALMesh)
    @with_kw immutable CALMesh
        coord::Matrix{Float64}
        dof::Matrix{Int}
        edof::Matrix{Int}
        topology::Matrix{Int}
        ex::Matrix{Float64}
        ey::Matrix{Float64}
        ez::Matrix{Float64}
        ndofs::Int
    end
end

nelems(cm::CALMesh) = size(cm.ex, 2)
ndofs(cm::CALMesh) = cm.ndofs

function get_freefixed(cm::CALMesh, bc)
    d_pres = convert(Vector{Int}, bc[:,1])   # prescribed dofs
    d_free = setdiff(cm.dof, d_pres) # free dofs
    return d_free, d_pres
end

function setup_geometry{dim}(mesh_file, nslips, grad_dofs, ::Type{Dim{dim}})
    mesh = ComsolMeshReader.read_mphtxt(mesh_file)
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

    ndofs_per_node = dim + grad_dofs * nslips
    dofs = reshape(collect(1:ndofs_per_node*n_nodes), (ndofs_per_node, n_nodes))

    Edof = zeros(Int, ndofs_per_node * nnodes, length(tri_elements))
    Ex = zeros(Float64, nnodes, length(tri_elements))
    Ey = zeros(Float64, nnodes, length(tri_elements))
    Ez = zeros(Float64, nnodes, length(tri_elements))


    for i in 1:length(tri_elements)
        verts = tri_elements[i]
        if dim == 2
            Edof[:, i] = [dofs[:,verts[1]]; dofs[:,verts[2]]; dofs[:,verts[3]]]
        elseif dim == 3
            Edof[:, i] = [dofs[:,verts[1]]; dofs[:,verts[2]]; dofs[:,verts[3]]; dofs[:,verts[4]]]
        end
        if dim == 2
            Ex[:,i] = [coordinates[verts[1]][1], coordinates[verts[2]][1], coordinates[verts[3]][1]]
            Ey[:,i] = [coordinates[verts[1]][2], coordinates[verts[2]][2], coordinates[verts[3]][2]]
        elseif dim == 3
            Ex[:,i] = [coordinates[verts[1]][1], coordinates[verts[2]][1], coordinates[verts[3]][1], coordinates[verts[4]][1]]
            Ey[:,i] = [coordinates[verts[1]][2], coordinates[verts[2]][2], coordinates[verts[3]][2], coordinates[verts[4]][2]]
            Ez[:,i] = [coordinates[verts[1]][3], coordinates[verts[2]][3], coordinates[verts[3]][3], coordinates[verts[4]][3]]
        end
    end

    Bu = Int[]
    Bg = Int[]
    for n in boundary_nodes
        append!(Bu, dofs[1:dim, n])
        append!(Bg, dofs[dim+1:end, n])
    end

    coords_mat = reinterpret(Float64, coordinates, (dim, length(coordinates)))

    mesh = CALMesh(coords_mat, dofs, Edof, topology, Ex, Ey, Ez, length(dofs))

   return mesh, Bu, Bg
end