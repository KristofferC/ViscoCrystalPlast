abstract AbstractBufferCollection

get_buffer{T <: GradientNumber}(buff_coll::AbstractBufferCollection, ::Type{T}) = buff_coll.buff_grad
get_buffer{T}(buff_coll::AbstractBufferCollection, ::Type{T}) = buff_coll.buff_float

function PrimalBuffer{Q, T}(problem::AbstractProblem, Tq::Type{Q}, Tt::Type{T})
    PrimalBuffer(create_buffer(problem, Tq), create_buffer(problem, Tt))
end
