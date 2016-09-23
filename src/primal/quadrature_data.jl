

type CrystPlastPrimalQD{dim, T, M} <: QuadratureData
    σ::SymmetricTensor{2, dim, T, M}
    ε::SymmetricTensor{2, dim, T, M}
    ε_p::SymmetricTensor{2, dim, T, M}
    τ_di::Vector{T}
    τ::Vector{T}
    ξo::Vector{T}
    ξ⟂::Vector{T}
end

function CrystPlastPrimalQD{dim}(nslip, ::Type{Dim{dim}})
    σ = zero(SymmetricTensor{2, dim})
    ε = zero(SymmetricTensor{2, dim})
    ε_p = zero(SymmetricTensor{2, dim})
    τ_di = zeros(nslip)
    τ = zeros(nslip)
    ξo = zeros(Float64, nslip)
    ξ⟂ = zeros(Float64, nslip)
    return CrystPlastPrimalQD(σ, ε, ε_p, τ_di, τ, ξo, ξ⟂)
end

type CrystPlastPrimalQDSerializer{dim, T, M}
    m::Matrix{Vector{Float64}}
end

function JLD.writeas{dim, T, M}(datam::Matrix{CrystPlastPrimalQD{dim, T, M}})
    mat = Matrix{Vector{Float64}}(size(datam))
    for i in 1:size(datam, 1), j in 1:size(datam, 2)
        data = datam[i, j]
        mat[i,j] = [vec(data.σ); vec(data.ε); vec(data.ε_p); data.τ_di; data.τ; data.ξo; data.ξ⟂]
    end
    CrystPlastPrimalQDSerializer{dim, T, M}(mat)
end



function JLD.readas{dim, T, M}(serdata::CrystPlastPrimalQDSerializer{dim, T, M})
    mat = serdata.m
    qdata = Matrix{CrystPlastPrimalQD{dim, T, M}}(size(mat))

    nslips = (length(mat[1,1]) - 3*dim*dim) ÷ 4
    tens_size = dim * dim

    for i in 1:size(qdata, 1), j in 1:size(qdata, 2)
        c_len = 1
        data = mat[i, j]
        σ = SymmetricTensor{2, dim}(data[c_len:c_len + tens_size - 1])
        c_len += tens_size
        ε = SymmetricTensor{2, dim}(data[c_len:c_len + tens_size - 1])
        c_len += tens_size
        ε_p = SymmetricTensor{2, dim}(data[c_len:c_len + tens_size - 1])
        c_len += tens_size
        τ_di = data[c_len:c_len + nslips - 1]
        c_len += nslips
        τ = data[c_len:c_len + nslips - 1]
        c_len += nslips
        ξo = data[c_len:c_len + nslips - 1]
        c_len += nslips
        ξ⟂ = data[c_len:c_len + nslips - 1]
        c_len += nslips
        qdata[i,j] = CrystPlastPrimalQD{dim, T, M}(σ, ε, ε_p, τ_di, τ, ξo, ξ⟂)
    end
    return qdata
end

get_type{dim, T, M}(::Type{CrystPlastPrimalQD{dim, T, M}}) = CrystPlastPrimalQD

function Base.:*(n::Number, qd::CrystPlastPrimalQD)
    CrystPlastPrimalQD(n * qd.σ, n * qd.ε, n * qd.ε_p, n * qd.τ_di, n * qd.τ, n * qd.ξo, n * qd.ξ⟂)
end

Base.:*(qd::CrystPlastPrimalQD, n::Number) = n * qd

function Base.:/(qd::CrystPlastPrimalQD, n::Number)
    CrystPlastPrimalQD(qd.σ / n, qd.ε / n, qd.ε_p / n, qd.τ_di / n, qd.τ / n, qd.ξo / n, qd.ξ⟂ / n)
end

function Base.:-(qd1::CrystPlastPrimalQD, qd2::CrystPlastPrimalQD)
    CrystPlastPrimalQD(qd1.σ - qd2.σ, qd1.ε - qd2.ε, qd1.ε_p - qd2.ε_p, qd1.τ_di - qd2.τ_di, qd1.τ - qd2.τ, qd1.ξo - qd2.ξo, qd1.ξ⟂ - qd2.ξ⟂)
end


function Base.:.*(qd1::CrystPlastPrimalQD, qd2::CrystPlastPrimalQD)
    CrystPlastPrimalQD(qd1.σ .* qd2.σ, qd1.ε .* qd2.ε, qd1.ε_p .* qd2.ε_p, qd1.τ_di .* qd2.τ_di, qd1.τ .* qd2.τ, qd1.ξo .* qd2.ξo, qd1.ξ⟂ .* qd2.ξ⟂)
end

function Base.:+(qd1::CrystPlastPrimalQD, qd2::CrystPlastPrimalQD)
    CrystPlastPrimalQD(qd1.σ + qd2.σ, qd1.ε + qd2.ε, qd1.ε_p + qd2.ε_p, qd1.τ_di + qd2.τ_di, qd1.τ + qd2.τ, qd1.ξo + qd2.ξo, qd1.ξ⟂ + qd2.ξ⟂)
end
