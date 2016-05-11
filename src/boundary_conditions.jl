immutable DirichletBoundaryConditions
    dof_ids::Vector{Int}
    dof_types::Vector{Symbol}
    values::Vector{Float64}
end

function Base.getindex(dbc::DirichletBoundaryConditions, field::Symbol)
    if !(field in dbc.dof_types)
        error("unknown symbol")
    end
    values = Float64[]
    for i in eachindex(dbc.dof_ids)
        if dbc.dof_types[i] == field
            push!(values, dbc.values[i])
        end
    end
    return values
end

set_value(dbc::DirichletBoundaryConditions, v, dof_id::Int) = dbc.values[dof_id] = v


function DirichletBoundaryConditions(dofs::Dofs, nodes::Vector{Int}, dbc_fields::Vector{Symbol})
    bc_dofs = findin(dbc_fields, dofs.dof_types)
    if length(bc_dofs) != length(dbc_fields)
        error("Some fields not found")
    end
    dof_types = Vector{Symbol}(length(bc_dofs) * length(nodes))
    dof_ids = Vector{Int}(length(bc_dofs) * length(nodes))
    count = 1
    for i in nodes
        for field in bc_dofs
            dof_types[count] = dbc_fields[field]
            dof_ids[count] = dofs.dof_ids[field, i]
            count += 1
        end
    end

    return DirichletBoundaryConditions(dof_ids, dof_types, zeros(Float64, size(dof_ids)))
end

function get_freefixed(dofs::Dofs, bc::DirichletBoundaryConditions)
    return setdiff(dofs.dof_ids, d_pres) # free dofs
end

function update_bcs!(mesh::GeometryMesh, dofs::Dofs, bc::DirichletBoundaryConditions, time::Float64, f)
    dofs_per_node = length(dofs.dof_types)
    for i in eachindex(bc.dof_ids)
        dof_type = bc.dof_types[i]
        dof_id = bc.dof_ids[i]
        node = div(dof_id + dofs_per_node -1, dofs_per_node)
        ViscoCrystalPlast.set_value(bc, f(dof_type, mesh.coords[:, node], time), i)
    end
end

apply!(v::Vector, bc::DirichletBoundaryConditions) = v[bc.dof_ids] = bc.values
