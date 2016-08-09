import Base.Order.Forward

function create_sparsity_pattern(dh::DofHandler)
    mesh = dh.mesh
    I = Int[]
    J = Int[]
    for element_id in 1:nelements(mesh)
        edof = dofs_element(dh, element_id)
        n_dofs = length(edof)
        for dof1 in 1:n_dofs, dof2 in 1:n_dofs
            push!(J, edof[dof1])
            push!(I, edof[dof2])
        end
    end
    V = zeros(length(I))
    K = sparse(I, J, V)
    return K
end

function create_lookup(A::SparseMatrixCSC, dh::DofHandler)
    mesh = dh.mesh
    dofs_per_ele = length(dofs_element(dh, 1))
    lookup_table = zeros(Int,  dofs_per_ele^2, nelements(mesh))
    for element_id in 1:nelements(mesh)
        k = 1
        edof = dofs_element(dh, element_id)
        n_dofs = length(edof)
        for i0 in 1:n_dofs, i1 in 1:n_dofs

            r1 = Int(A.colptr[edof[i1]])
            r2 = Int(A.colptr[edof[i1]+1]-1)
            i = (r1 > r2) ? r1 : searchsortedfirst(A.rowval, edof[i0], r1, r2, Forward)
            lookup_table[k, element_id] = i
            k += 1
        end
    end
    return lookup_table
end

function assemble!(K::SparseMatrixCSC, Ke::Matrix, element::Int, lookup_table::Matrix{Int})
    k = 1
    ndofs_per_ele = Int(sqrt(size(lookup_table, 1)))
    @inbounds for loc1 in 1:ndofs_per_ele, loc2 in 1:ndofs_per_ele
        lin_idx = lookup_table[k, element]
        k += 1
        K.nzval[lin_idx] += Ke[loc1, loc2]
    end
end

function assemble!(K::SparseMatrixCSC, Ke::Matrix, dofs::AbstractVector)
    for (loc1, glob1) in enumerate(dofs)
        for (loc2, glob2) in enumerate(dofs)
            add!(K, Ke[loc1, loc2], glob1, glob2)
        end
    end
end

function assemble!(f::Vector{Float64}, fe::Vector{Float64}, dofs::AbstractVector)
    for (loc, glob) in enumerate(dofs)
        f[glob] += fe[loc]
    end
end

function mutateindex!{T,Ti}(A::SparseMatrixCSC{T,Ti}, f, i0::Integer, i1::Integer)
    m = A.m
    n = A.n
    if !(1 <= i0 <= A.m && 1 <= i1 <= n); throw(BoundsError()); end
    r1 = Int(A.colptr[i1])
    r2 = Int(A.colptr[i1+1]-1)
    i = (r1 > r2) ? r1 : searchsortedfirst(A.rowval, i0, r1, r2, Forward)

    if (i <= r2) && (A.rowval[i] == i0)
        A.nzval[i] = f(A.nzval[i])
    else
        error("Nonexisting value")
    end
    return A
end

add!(A::AbstractMatrix, v, i0::Integer, i1::Integer) = mutateindex!(A, (i) -> i + v, i0, i1)
