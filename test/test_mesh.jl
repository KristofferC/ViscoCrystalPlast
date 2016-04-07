import ViscoCrystalPlast: create_mesh, element_coordinates, add_dofs, dofs_node, dofs_element

import ViscoCrystalPlast: DirichletBoundaryConditions

mesh = create_mesh("test_mesh.mphtxt")

@test size(mesh.coords) == (2, 286)
@test mesh.coords[:, 6] ≈ [0.019279820415859801, 0.021354516264327765]

# 4 3 9
@test element_coordinates(mesh, 5) ≈
        [mesh.coords[:, 4] mesh.coords[:, 3] mesh.coords[:, 9]]

dofs = add_dofs(mesh, [:u, :v])

@test dofs_node(mesh, dofs, 2) == [3, 4]


# 1 7 6
dofs_ele = dofs_element(mesh, dofs, 2)


@test length(dofs_ele) == 6
@test dofs_ele == vec([dofs_node(mesh, dofs, 1) dofs_node(mesh, dofs, 7) dofs_node(mesh, dofs, 6)])

dbc_u = DirichletBoundaryConditions(dofs, mesh.boundary_nodes, [:u])
@test length(dbc_u.dof_ids) == length(mesh.boundary_nodes)

dbc_uv = DirichletBoundaryConditions(dofs, mesh.boundary_nodes, [:u, :v])
@test length(dbc_uv.dof_ids) == 2*length(mesh.boundary_nodes)