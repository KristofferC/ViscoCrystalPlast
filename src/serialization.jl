function serialize!(data, Type)
    for field in fieldnames(Type)
        serialize!(data, getfield(Type, field))
    end
    if length(fieldnames(Type)) == 0
        error("Unhandled $Type")
    end
    return data
end

function serialize!(data, v::Number)
    append!(data, v)
end

function serialize!(data, v::AbstractTensor)
    append!(data, v.data)
end

function serialize!{T}(data, v::AbstractVecOrMat{T})
    if T <: Number
        append!(data, vec(v))
    else
        for ele in v
            serialize!(data, ele)
        end
    end
end

scalar(v, c_len) = (val = v[c_len]; c_len += 1; return val, c_len)

function vector(v, T::Type, len::Int, n::Int, c_len::Int)
    val = reinterpret(T, v[c_len: c_len + n * len - 1], (n,))
    c_len += n * len
    return val, c_len
end
