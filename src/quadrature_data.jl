immutable QuadratureData{T}
    temp_data::T
    converged_data::T
end

temp_data(qd::QuadratureData) = qd.temp_data
converged_data(qd::QuadratureData) = qd.converged_data
