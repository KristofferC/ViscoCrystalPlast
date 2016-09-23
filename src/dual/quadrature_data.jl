type CrystPlastDualQD{dim, T, M} <: QuadratureData
    σ::SymmetricTensor{2, dim, T, M}
    ε::SymmetricTensor{2, dim, T, M}
    ε_p::SymmetricTensor{2, dim, T, M}
    τ_di::Vector{T}
    τ::Vector{T}
    γ::Vector{T}
    χ⟂::Vector{T}
    χo::Vector{T}
end

function CrystPlastDualQD{dim}(nslip, ::Type{Dim{dim}})
    σ = zero(SymmetricTensor{2, dim})
    ε = zero(SymmetricTensor{2, dim})
    ε_p = zero(SymmetricTensor{2, dim})
    τ_di = zeros(nslip)
    τ = zeros(nslip)
    γ = zeros(nslip)
    χ⟂ = zeros(nslip)
    χo = zeros(nslip)
    return CrystPlastDualQD(σ, ε, ε_p, τ_di, τ, γ, χ⟂, χo)
end


type CrystPlastDualQDSerializer{dim, T, M}
    m::Matrix{Vector{Float64}}
end

function JLD.writeas{dim, T, M}(datam::Matrix{CrystPlastDualQD{dim, T, M}})
    mat = Matrix{Vector{Float64}}(size(datam))
    for i in 1:size(datam, 1), j in 1:size(datam, 2)
        data = datam[i, j]
        mat[i,j] = [vec(data.σ); vec(data.ε); vec(data.ε_p); data.τ_di; data.τ; data.γ; data.χ⟂; data.χo]
    end
    CrystPlastDualQDSerializer{dim, T, M}(mat)
end



function JLD.readas{dim, T, M}(serdata::CrystPlastDualQDSerializer{dim, T, M})
    mat = serdata.m
    qdata = Matrix{CrystPlastDualQD{dim, T, M}}(size(mat))

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
        γ = data[c_len:c_len + nslips - 1]
        c_len += nslips
        χ⟂ = data[c_len:c_len + nslips - 1]
        c_len += nslips
        χo = data[c_len:c_len + nslips - 1]
        c_len += nslips
        qdata[i,j] = CrystPlastDualQD{dim, T, M}(σ, ε, ε_p, τ_di, τ, γ, χ⟂, χo)
    end
    return qdata
end


get_type{dim, T, M}(::Type{CrystPlastDualQD{dim, T, M}}) = CrystPlastDualQD

function Base.:*(n::Number, qd::CrystPlastDualQD)
    CrystPlastDualQD(n * qd.σ, n * qd.ε, n * qd.ε_p, n * qd.τ_di, n * qd.τ, n * qd.γ, n * qd.χ⟂, n * qd.χo)
end

Base.:*(qd::CrystPlastDualQD, n::Number) = n * qd

function Base.:/(qd::CrystPlastDualQD, n::Number)
    CrystPlastDualQD(qd.σ / n, qd.ε / n, qd.ε_p / n, qd.τ_di / n, qd.τ / n, qd.γ / n, qd.χ⟂ / n, qd.χo / n)
end

function Base.:-(qd1::CrystPlastDualQD, qd2::CrystPlastDualQD)
    CrystPlastDualQD(qd1.σ - qd2.σ, qd1.ε - qd2.ε, qd1.ε_p - qd2.ε_p, qd1.τ_di - qd2.τ_di, qd1.τ - qd2.τ, qd1.γ - qd2.γ, qd1.χ⟂ - qd2.χ⟂, qd1.χo - qd2.χo)
end

function Base.:.*(qd1::CrystPlastDualQD, qd2::CrystPlastDualQD)
    CrystPlastDualQD(qd1.σ .* qd2.σ, qd1.ε .* qd2.ε, qd1.ε_p .* qd2.ε_p, qd1.τ_di .* qd2.τ_di, qd1.τ .* qd2.τ, qd1.γ .* qd2.γ, qd1.χ⟂ .* qd2.χ⟂,  qd1.χo .* qd2.χo)
end

function Base.:+(qd1::CrystPlastDualQD, qd2::CrystPlastDualQD)
    CrystPlastDualQD(qd1.σ + qd2.σ, qd1.ε + qd2.ε, qd1.ε_p + qd2.ε_p, qd1.τ_di + qd2.τ_di, qd1.τ + qd2.τ, qd1.γ + qd2.γ, qd1.χ⟂ + qd2.χ⟂, qd1.χo + qd2.χo)
end


function create_quadrature_data{dim}(::DualProblem, ::Type{Dim{dim}}, quad_rule, nslip, n_elements)
    n_qpoints = length(points(quad_rule))
    mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:n_elements]
    temp_mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:n_elements]
    QuadratureData(mss, temp_mss)
end
