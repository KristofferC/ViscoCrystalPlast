#function u_dofs(dofs_u, nnodes, dofs_g, nslip)
#    dofs = Int[]
#    count = 0
#    for i in 1:nnodes
#        for j in 1:dofs_u
#            push!(dofs, count+j)
#        end
#        count += nslip * dofs_g + dofs_u
#    end
#    return dofs
#end

compute_udofs(dofs_u, nnodes, dofs_g, nslip) = 1:nnodes*dofs_u

function compute_γdofs(dofs_u, nnodes, dofs_g, nslip, slip)
    start = nnodes * dofs_u + 1 + (slip - 1) * nnodes * dofs_g
    return start : start + nnodes * dofs_g - 1
end



#function γ_dofs(dofs_u, nnodes, dofs_g, nslip, slip)
#    dofs = Int[]
#    count = dofs_u + (slip -1) * dofs_g
#    for i in 1:nnodes
#        for j in 1:dofs_g
#            push!(dofs, count+j)
#        end
#        count += dofs_u + nslip * dofs_g
#    end
#    return dofs
#end


function compute_ξdofs(dofs_u, nnodes, dofs_g, nslip, slip, ξtype::Symbol)
    @assert slip <= nslip
    u_offset = nnodes * dofs_u
    length = nnodes * dofs_g
    ξ_offset = (slip - 1) * nnodes

    if ξtype == :ξ⟂
        offset = 0
    elseif ξtype == :ξo
        offset = nslip  * nnodes
    else
        error("unknown ξ type")
    end
    tot_offset = u_offset + ξ_offset + offset
    return tot_offset + 1 : tot_offset + length
end

#ξ_dofs(3, 4, 1, 2, 1, :ξo)


function extract!(a::AbstractArray, b::AbstractArray, c::AbstractArray)
    @assert length(c) == length(a)
    for (i,j) in enumerate(c)
        a[i] = b[j]
    end
end
