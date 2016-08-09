function move_quadrature_data_to_nodes{QD <: QuadratureData, dim}(quad_data::AbstractVecOrMat{QD}, mesh, quad_rule::QuadratureRule{dim})
    nslips = length(quad_data[1,1].τ)

    quad_data_nodes = [get_type(QD)(nslips, Dim{dim}) for i = 1:nnodes(mesh)]
    count_nodes = zeros(Int, nnodes(mesh))
    for i in 1:nelements(mesh)
        for q_point in 1:length(points(quad_rule))
            for node in element_vertices(mesh, i)
                count_nodes[node] += 1
                quad_data_nodes[node] += quad_data[q_point, i]
            end
        end
    end

    for i in 1:nnodes(mesh)
        quad_data_nodes[i] /= count_nodes[i]
    end

    return quad_data_nodes
end


#=
typealias TrigType GeometricalPredicates.UnOrientedTriangle{GeometricalPredicates.Point2D}

function get_global_gauss_point_coordinates{dim, T}(fe_values::FEValues{dim, T}, mesh)
    n_qpoints = length(points(get_quadrule(fe_values)))
    n_elements = size(mesh.topology, 2)
    n_basefuncs = n_basefunctions(get_functionspace(fe_values))
    x_glob = Matrix{Vec{dim, T}}(n_qpoints, n_elements)
    e_coordinates = zeros(dim,  n_basefuncs)
    for element_id in 1:n_elements
        element_coordinates!(e_coordinates , mesh, element_id)
        x_vec = reinterpret(Vec{dim, T}, e_coordinates, (n_basefuncs,))
        reinit!(fe_values, x_vec)
        for q_point in 1:n_qpoints
            x_glob[q_point, element_id] = function_vector_value(fe_values, q_point, x_vec)
        end
    end
    return x_glob
end

function find_bounding_element_to_gps(fine_gp_coords, coarse_mesh)
    dim = 2
    n_coarse_elements = size(coarse_mesh.topology, 2)
    fine_points = reinterpret(GeometricalPredicates.Point2D, fine_gp_coords, (size(fine_gp_coords,2),))
    coarse_points = reinterpret(GeometricalPredicates.Point2D, coarse_mesh.coords, (size(coarse_mesh.coords,2),))
    primitives = TrigType[Primitive(coarse_points[coarse_mesh.topology[:, i]]...) for i in 1:n_coarse_elements]

    coarse_centers = zeros(dim, n_coarse_elements)
    for e in 1:n_coarse_elements
        center = mean(ViscoCrystalPlast.element_coordinates(coarse_mesh, e), 2)
        coarse_centers[:, e] = center # Point(center[1], center[2])
    end
    coarse_center_tree = KDTree(coarse_centers)
    elements = Int[]

    surrounding_elements = Int[]
    for q_point in 1:length(fine_gp_coords)
        c = vec(fine_gp_coords[q_point])
        # Find 10 closest elements to the fine node.. should be enough
        candidates, dist = knn(coarse_center_tree, c, 10, true)
        found_ele = false
        for element in candidates
            if check_in_triangle(primitives[element], fine_points[q_point])
                found_ele = true
                push!(surrounding_elements, element)
                break
            end
        end
        if found_ele == false
            error("Did not find triangle for node $node")
        end
    end
    return surrounding_elements
end




function interpolate_to{QD <: QuadratureData}(solution_coarse_nodes, quad_data_coarse::AbstractVector{QD}, coarse_mesh,
                                              fine_mesh, dofs_coarse, dofs_fine, gp_coords, func_space)
    dim = 2
    nslips = length(quad_data_coarse[1].τ)
    tot_nodes = size(fine_mesh.coords, 2)
    n_coarse_elements = size(coarse_mesh.topology, 2)

    solution_fine_nodes = zeros(length(dofs_fine.dof_ids))

    fine_points = reinterpret(GeometricalPredicates.Point2D, fine_mesh.coords, (size(fine_mesh.coords,2),))
    coarse_points = reinterpret(GeometricalPredicates.Point2D, coarse_mesh.coords, (size(coarse_mesh.coords,2),))
    primitives = TrigType[Primitive(coarse_points[coarse_mesh.topology[:, i]]...) for i in 1:n_coarse_elements]

    quad_data_nodes = [get_type(QD)(nslips, Dim{dim}) for i = 1:tot_nodes]

    coarse_centers = zeros(dim, n_coarse_elements)
    for e in 1:n_coarse_elements
        center = mean(ViscoCrystalPlast.element_coordinates(coarse_mesh, e), 2)
        coarse_centers[:, e] = center # Point(center[1], center[2])
    end
    coarse_center_tree = KDTree(coarse_centers)

    # Find 10 closest elements to the fine node.. should be enough
    found_ele = false
    for gp in eachindex(gp_coords)
        c = fine_mesh.coords[:, node]
        candidates, dist = knn(coarse_center_tree, c , 10, true)
        found_ele = false
        for element in candidates
            if check_in_triangle(primitives[element], fine_points[node])
                found_ele = true
                sol_node, quad_data_node = interpolate(solution_coarse_nodes, quad_data_coarse, coarse_mesh, element, c, func_space, dofs_coarse)
                solution_fine_nodes[dofs_fine.dof_ids[:, node]] = sol_node
                quad_data_nodes[node] = quad_data_node
                break
            end
        end
        if found_ele == false
            error("Did not find triangle for node $node")
        end
    end
    return solution_fine_nodes, quad_data_nodes
end


# Interpolates finite element values from a mesh to another set of points
function interpolate_to{QD <: QuadratureData}(solution_coarse_nodes, quad_data_coarse::AbstractVector{QD}, coarse_mesh,
                                              fine_mesh, dofs_coarse, dofs_fine, func_space)
    dim = 2
    nslips = length(quad_data_coarse[1].τ)
    tot_nodes = size(fine_mesh.coords, 2)
    n_coarse_elements = size(coarse_mesh.topology, 2)

    solution_fine_nodes = zeros(length(dofs_fine.dof_ids))

    fine_points = reinterpret(GeometricalPredicates.Point2D, fine_mesh.coords, (size(fine_mesh.coords,2),))
    coarse_points = reinterpret(GeometricalPredicates.Point2D, coarse_mesh.coords, (size(coarse_mesh.coords,2),))
    primitives = TrigType[Primitive(coarse_points[coarse_mesh.topology[:, i]]...) for i in 1:n_coarse_elements]

    quad_data_nodes = [get_type(QD)(nslips, Dim{dim}) for i = 1:tot_nodes]

    coarse_centers = zeros(dim, n_coarse_elements)
    for e in 1:n_coarse_elements
        center = mean(ViscoCrystalPlast.element_coordinates(coarse_mesh, e), 2)
        coarse_centers[:, e] = center # Point(center[1], center[2])
    end
    coarse_center_tree = KDTree(coarse_centers)

    # Find 10 closest elements to the fine node.. should be enough
    found_ele = false
    @time for node in 1:size(fine_mesh.coords, 2)
        c = fine_mesh.coords[:, node]
        candidates, dist = knn(coarse_center_tree, c , 10, true)
        found_ele = false
        for element in candidates
            if check_in_triangle(primitives[element], fine_points[node])
                found_ele = true
                sol_node, quad_data_node = interpolate(solution_coarse_nodes, quad_data_coarse, coarse_mesh, element, c, func_space, dofs_coarse)
                solution_fine_nodes[dofs_fine.dof_ids[:, node]] = sol_node
                quad_data_nodes[node] = quad_data_node
                break
            end
        end
        if found_ele == false
            error("Did not find triangle for node $node")
        end
    end
    return solution_fine_nodes, quad_data_nodes
end



# Computes the value from quadrature data in nodes and nodal field at a given element
function interpolate{QD <: QuadratureData}(solution_coarse_nodes, quad_data_coarse::AbstractVector{QD}, coarse_mesh, element, global_coord, func_space, dofs_coarse)
    dim = 2
    nslips = length(quad_data_coarse[1].τ)
    n_basefuncs = n_basefunctions(func_space)
    verts = coarse_mesh.topology[:, element]
    vert_coords = coarse_mesh.coords[:, verts]
    ele_quad_data = QD[quad_data_coarse[i] for i in verts]
    global_coord_vec = Vec{2}((global_coord[1], global_coord[2]))

    verts_vec = reinterpret(Vec{2, Float64}, vert_coords, (size(vert_coords,2),))
    ξ = find_local(func_space, verts_vec, global_coord_vec)
    ele_dofs = dofs_element(coarse_mesh, dofs_coarse, element)
    u_ele = solution_coarse_nodes[ele_dofs]
    dofs_per_node = div(length(ele_dofs), n_basefuncs)
    u_coarse_nodes = [u_ele[i:i+dofs_per_node-1] for i in 1:dofs_per_node:length(u_ele)]

    N = JuAFEM.value(func_space, ξ)
    u_point = zero(u_coarse_nodes[1])
    for j in 1:n_basefuncs
        u_point += N[j] * u_coarse_nodes[j]
    end

    quad_data_node = get_type(QD)(nslips, Dim{dim})
    for j in 1:n_basefuncs
        quad_data_node += N[j] * ele_quad_data[j]
    end

    return u_point, quad_data_node
end


# Find the local coordinate given a global coordinate
function find_local{dim, T}(func_space, vertices::Vector{Vec{dim, T}}, global_coord)
    local_guess = zero(Vec{2})
    n_basefuncs = n_basefunctions(func_space)

    for iter in 1:10
        if iter == 10
            error("woot m8")
        end

        N = JuAFEM.value(func_space, local_guess)
        global_guess = zero(Vec{2})
        for j in 1:n_basefuncs
            global_guess += N[j] * vertices[j]
        end

        residual = global_guess - global_coord

        if norm(residual) <= 1e-10
            break
        end

        dNdξ = JuAFEM.derivative(func_space, local_guess)

        J = zero(Tensor{2, 2})
        for j in 1:n_basefuncs
            J += vertices[j] ⊗ dNdξ[j]
        end

        local_guess -= inv(J) ⋅ residual
    end

    return local_guess
end

function check_in_triangle(t, p)
    p0x = getx(t._a)
    p1x = getx(t._b)
    p2x = getx(t._c)

    p0y = gety(t._a)
    p1y = gety(t._b)
    p2y = gety(t._c)

    px = getx(p)
    py = gety(p)

    s = p0y*p2x - p0x*p2y + (p2y - p0y)*px + (p0x - p2x)*py
    t = p0x*p1y - p0y*p1x + (p0y - p1y)*px + (p1x - p0x)*py

    if s >= 0.0 && t >= 0.0 && 1-s-t >= 0.0
        return true
    else
        return false
    end
end


function interpolate_to_quads{QD <: QuadratureData}(solution_coarse_nodes, quad_data_coarse::AbstractVector{QD}, coarse_mesh,
                                                    fine_mesh, dofs_coarse, dofs_fine, gp_coords, func_space)
    dim = 2
    n_qpoints = length(points(get_quadrule(fe_values)))
    n_elements = size(mesh.topology, 2)
    n_basefuncs = n_basefunctions(get_functionspace(fe_values))
    nslips = length(quad_data_coarse[1].τ)
    tot_nodes = size(fine_mesh.coords, 2)
    n_coarse_elements = size(coarse_mesh.topology, 2)

    solution_fine_nodes = zeros(length(dofs_fine.dof_ids))

    fine_points = reinterpret(GeometricalPredicates.Point2D, fine_mesh.coords, (size(fine_mesh.coords,2),))
    coarse_points = reinterpret(GeometricalPredicates.Point2D, coarse_mesh.coords, (size(coarse_mesh.coords,2),))
    primitives = TrigType[Primitive(coarse_points[coarse_mesh.topology[:, i]]...) for i in 1:n_coarse_elements]

    quad_data_nodes = [get_type(QD)(nslips, Dim{dim}) for i = 1:tot_nodes]

    coarse_centers = zeros(dim, n_coarse_elements)
    for e in 1:n_coarse_elements
        center = mean(ViscoCrystalPlast.element_coordinates(coarse_mesh, e), 2)
        coarse_centers[:, e] = center # Point(center[1], center[2])
    end
    coarse_center_tree = KDTree(coarse_centers)

    # Find 10 closest elements to the fine node.. should be enough
    found_ele = false
    for element_id in 1:n_elements
        for q_point in 1:n_qpoints
            c = vec(x_glob[q_point, element_id])
            candidates, dist = knn(coarse_center_tree, c , 10, true)
            found_ele = false
            for element in candidates
                if check_in_triangle(primitives[element], fine_points[node])
                    found_ele = true
                    println("found it: $element")
                    sol_node, quad_data_node = interpolate(solution_coarse_nodes, quad_data_coarse, coarse_mesh, element, c, func_space, dofs_coarse)
                    solution_fine_nodes[dofs_fine.dof_ids[:, node]] = sol_node
                    quad_data_nodes[node] = quad_data_node
                    break
                end
            end
        if found_ele == false
            error("Did not find triangle for node $node")
        end
    end
    return solution_fine_nodes, quad_data_nodes
end
=#