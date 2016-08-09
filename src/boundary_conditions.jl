immutable DirichletBoundaryCondition
    f::Function
    nodes::Vector{Int}
    field::Symbol
    components::Vector{Int}
    idxoffset::Int
end


immutable DirichletBoundaryConditions
    bcs::Vector{DirichletBoundaryCondition}
    dofs::Vector{Int}
    values::Vector{Float64}
    dh::DofHandler
    closed::Ref{Bool}
end

function DirichletBoundaryConditions(dh::DofHandler)
    @dbg_assert isclosed(dh)
    DirichletBoundaryConditions(DirichletBoundaryCondition[], Int[], Float64[], dh, Ref(false))
end


dirichlet_dofs(dbcs::DirichletBoundaryConditions) = dbcs.dofs
free_dofs(dbcs::DirichletBoundaryConditions) = setdiff(dbcs.dh.dofs_nodes, dbcs.dofs)
function close!(dbcs::DirichletBoundaryConditions)
    fill!(dbcs.values, NaN)
    dbcs.closed[] = true
end

function add_dirichletbc!(dbcs::DirichletBoundaryConditions, field::Symbol,
                          nodes::Vector{Int}, f::Function, component::Int=1)
    add_dirichletbc!(dbcs, field, nodes, f, [component])
end

function add_dirichletbc!(dbcs::DirichletBoundaryConditions, field::Symbol,
                          nodes::Vector{Int}, f::Function, components::Vector{Int})
    @dbg_assert field in dbcs.dh.field_names
    for component in components
        @dbg_assert 0 < component <= ndim(dbcs.dh, field)
    end

    dofs_bc = Int[]
    offset = dof_offset(dbcs.dh, field)
    for node in nodes
        for component in components
            push!(dofs_bc, dbcs.dh.dofs_nodes[offset + component, node])
        end
    end

    n_bcdofs = length(dofs_bc)

    append!(dbcs.dofs, dofs_bc)
    idxoffset = length(dbcs.values)
    resize!(dbcs.values, length(dbcs.values) + n_bcdofs)

    push!(dbcs.bcs, DirichletBoundaryCondition(f, nodes, field, components, idxoffset))

end

function update_dirichletbcs!(dbcs::DirichletBoundaryConditions, time::Float64 = 0.0)
    @dbg_assert dbcs.closed[]
    bc_offset = 0
    for dbc in dbcs.bcs
        # Function barrier
        _update_dirichletbcs!(dbcs.values, dbc.f, dbc.nodes, dbc.field,
                              dbc.components, dbcs.dh, dbc.idxoffset, time)
    end
end

function _update_dirichletbcs!(values::Vector{Float64}, f::Function, nodes::Vector{Int}, field::Symbol,
                              components::Vector{Int}, dh::DofHandler, idx_offset::Int, time::Float64)
    mesh = dh.mesh
    offset = dof_offset(dh, field)
    current_dof = 1
     for node in nodes
        x = node_coordinates(mesh, node)
        bc_value = f(x, time)
        @dbg_assert length(bc_value) == length(components)
        for i in 1:length(components)
            values[current_dof + idx_offset] = bc_value[i]
            current_dof += 1
        end
    end
end

function vtk_point_data(vtkfile, dbcs::DirichletBoundaryConditions)
    unique_fields = []
    for dbc in dbcs.bcs
        push!(unique_fields, dbc.field)
    end
    unique_fields = unique(unique_fields)

    for field in unique_fields
        nd = ndim(dbcs.dh, field)
        data = zeros(Float64, nd, length(dbcs.dh.mesh.coords))
        for dbc in dbcs.bcs
            if dbc.field != field
                continue
            end

            for node in dbc.nodes
                for component in dbc.components
                    data[component, node] = 1.0
                end
            end
        end
        vtk_point_data(vtkfile, data, string(field)*"_bc")
    end
    return vtkfile
end



apply!(v::Vector, bc::DirichletBoundaryConditions) = v[bc.dofs] = bc.values

#=
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



function update_bcs!(mesh::GeometryMesh, dofs::Dofs, bc::DirichletBoundaryConditions, time::Float64, f)
    dofs_per_node = length(dofs.dof_types)
    for i in eachindex(bc.dof_ids)
        dof_type = bc.dof_types[i]
        dof_id = bc.dof_ids[i]
        node = div(dof_id + dofs_per_node -1, dofs_per_node)
        set_value(bc, f(dof_type, mesh.coords[:, node], time), i)
    end
end


=#
