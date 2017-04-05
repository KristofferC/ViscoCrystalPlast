type CrystPlastPrimalQD{dim, T, M} <: QuadratureData
    σ::SymmetricTensor{2, dim, T, M}
    ε::SymmetricTensor{2, dim, T, M}
    ε_p::SymmetricTensor{2, dim, T, M}
    τ_di::Vector{T}
    τ::Vector{T}
    ξo::Vector{T}
    ξ⟂::Vector{T}
    γ::Vector{T}
end

function CrystPlastPrimalQD{dim}(nslip, ::Type{Dim{dim}})
    σ = zero(SymmetricTensor{2, dim})
    ε = zero(SymmetricTensor{2, dim})
    ε_p = zero(SymmetricTensor{2, dim})
    τ_di = zeros(nslip)
    τ = zeros(nslip)
    ξo = zeros(Float64, nslip)
    ξ⟂ = zeros(Float64, nslip)
    γ = zeros(Float64, nslip)
    return CrystPlastPrimalQD(σ, ε, ε_p, τ_di, τ, ξo, ξ⟂, γ)
end

get_type{dim, T, M}(::Type{CrystPlastPrimalQD{dim, T, M}}) = CrystPlastPrimalQD

function Base.:*(n::Number, qd::CrystPlastPrimalQD)
    CrystPlastPrimalQD(n * qd.σ, n * qd.ε, n * qd.ε_p, n * qd.τ_di, n * qd.τ, n * qd.ξo, n * qd.ξ⟂, n * qd.γ)
end

Base.:*(qd::CrystPlastPrimalQD, n::Number) = n * qd

function Base.:/(qd::CrystPlastPrimalQD, n::Number)
    CrystPlastPrimalQD(qd.σ / n, qd.ε / n, qd.ε_p / n, qd.τ_di / n, qd.τ / n, qd.ξo / n, qd.ξ⟂ / n, qd.γ / n)
end

function Base.:-(qd1::CrystPlastPrimalQD, qd2::CrystPlastPrimalQD)
    CrystPlastPrimalQD(qd1.σ - qd2.σ, qd1.ε - qd2.ε, qd1.ε_p - qd2.ε_p, qd1.τ_di - qd2.τ_di, qd1.τ - qd2.τ, qd1.ξo - qd2.ξo, qd1.ξ⟂ - qd2.ξ⟂, qd1.γ - qd2.γ)
end


function Base.:.*(qd1::CrystPlastPrimalQD, qd2::CrystPlastPrimalQD)
    CrystPlastPrimalQD(qd1.σ .* qd2.σ, qd1.ε .* qd2.ε, qd1.ε_p .* qd2.ε_p, qd1.τ_di .* qd2.τ_di, qd1.τ .* qd2.τ, qd1.ξo .* qd2.ξo, qd1.ξ⟂ .* qd2.ξ⟂, qd1.γ .* qd2.γ)
end

function Base.:+(qd1::CrystPlastPrimalQD, qd2::CrystPlastPrimalQD)
    CrystPlastPrimalQD(qd1.σ + qd2.σ, qd1.ε + qd2.ε, qd1.ε_p + qd2.ε_p, qd1.τ_di + qd2.τ_di, qd1.τ + qd2.τ, qd1.ξo + qd2.ξo, qd1.ξ⟂ + qd2.ξ⟂, qd1.γ + qd2.γ)
end
