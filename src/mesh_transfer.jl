# Transfers quadrature data given in qudarature points to nodes.

function move_quadrature_data_to_nodes{QD <: QuadratureData, dim}(quad_data::AbstractVecOrMat{QD}, mesh, quad_rule::QuadratureRule{dim})
    nslips = length(quad_data[1,1].τ)

    quad_data_nodes = [get_type(QD)(nslips, Dim{dim}) for i = 1:getnnodes(mesh)]
    count_nodes = zeros(Int, getnnodes(mesh))
    for i in 1:getncells(mesh)
        for q_point in 1:length(getpoints(quad_rule))
            for node in getcells(mesh, i).nodes
                count_nodes[node] += 1
                quad_data_nodes[node] += quad_data[q_point, i]
            end
        end
    end

    for i in 1:getnnodes(mesh)
        quad_data_nodes[i] /= count_nodes[i]
    end

    return quad_data_nodes
end