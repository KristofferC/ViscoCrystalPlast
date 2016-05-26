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

function γ_dofs(dofs_u, nnodes, dofs_g, nslip, slip)
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


function ξ_dofs(dofs_u, nnodes, dofs_g, nslip, slip, ξtype::Symbol)
    if ξtype == :ξo
        offset = 1
    elseif ξtype == :ξ⟂
        offset = 0
    else
        error("unknown ξ type")
    end
    dofs = Int[]
    count = dofs_u + (slip -1) * dofs_g
    for i in 1:nnodes
        push!(dofs, count+offset + 1)
        count += dofs_u + nslip * dofs_g
    end
    return dofs
end
