#k indicates last time step...
type CrystPlastDualQD{dim, T, M} <: QuadratureData
    σ::SymmetricTensor{2, dim, T, M}
    ε::SymmetricTensor{2, dim, T, M}
    ε_p::SymmetricTensor{2, dim, T, M}
    τ_di::Vector{T}
    τ::Vector{T}
    γ::Vector{T}
    χ⟂::Vector{T}
    χo::Vector{T}
    ψek::T
    ψgk::T
    ψe::T
    ψg::T
    χ_π::T
    ϕ_π::T
    τ_π::T
    π::T
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
    o = 0.0
    return CrystPlastDualQD(σ, ε, ε_p, τ_di, τ, γ, χ⟂, χo,
    o, o, o, o, o, o, o, o)
end

function Base.copy(qd::CrystPlastDualQD)
    CrystPlastDualQD(qd.σ, qd.ε, qd.ε_p, copy(qd.τ_di), copy(qd.τ), copy(qd.γ), copy(qd.χ⟂), copy(qd.χo),
                     qd.ψek, qd.ψgk, qd.ψe, qd.ψg, qd.χ_π, qd.ϕ_π, qd.τ_π, qd.π)
end

get_type{dim, T, M}(::Type{CrystPlastDualQD{dim, T, M}}) = CrystPlastDualQD

function Base.:*(n::Number, qd::CrystPlastDualQD)
    CrystPlastDualQD(n * qd.σ, n * qd.ε, n * qd.ε_p, n * qd.τ_di, n * qd.τ, n * qd.γ, n * qd.χ⟂, n * qd.χo,
                     n * qd.ψek, n * qd.ψgk, n * qd.ψe, n * qd.ψg, n * qd.χ_π, n * qd.ϕ_π , n * qd.τ_π, n * qd.π)
end

Base.:*(qd::CrystPlastDualQD, n::Number) = n * qd

function Base.:/(qd::CrystPlastDualQD, n::Number)
    CrystPlastDualQD(qd.σ / n, qd.ε / n, qd.ε_p / n, qd.τ_di / n, qd.τ / n, qd.γ / n, qd.χ⟂ / n, qd.χo / n,
                      qd.ψek / n,  qd.ψgk / n,  qd.ψe / n,  qd.ψg / n,  qd.χ_π / n,  qd.ϕ_π / n,  qd.τ_π / n,  qd.π / n)
end

function Base.:-(qd1::CrystPlastDualQD, qd2::CrystPlastDualQD)
    CrystPlastDualQD(qd1.σ - qd2.σ, qd1.ε - qd2.ε, qd1.ε_p - qd2.ε_p, qd1.τ_di - qd2.τ_di, qd1.τ - qd2.τ, qd1.γ - qd2.γ, qd1.χ⟂ - qd2.χ⟂, qd1.χo - qd2.χo,
                     qd1.ψek - qd2.ψek, qd1.ψgk - qd2.ψgk, qd1.ψe - qd2.ψe, qd1.ψg - qd2.ψg, qd1.χ_π - qd2.χ_π, qd1.ϕ_π - qd2.ϕ_π , qd1.τ_π - qd2.τ_π, qd1.π - qd2.π)
end

function Base.:.*(qd1::CrystPlastDualQD, qd2::CrystPlastDualQD)
    CrystPlastDualQD(qd1.σ .* qd2.σ, qd1.ε .* qd2.ε, qd1.ε_p .* qd2.ε_p, qd1.τ_di .* qd2.τ_di, qd1.τ .* qd2.τ, qd1.γ .* qd2.γ, qd1.χ⟂ .* qd2.χ⟂,  qd1.χo .* qd2.χo,
                     qd1.ψek .* qd2.ψek, qd1.ψgk .* qd2.ψgk, qd1.ψe .* qd2.ψe, qd1.ψg .* qd2.ψg, qd1.χ_π .* qd2.χ_π, qd1.ϕ_π .* qd2.ϕ_π , qd1.τ_π .* qd2.τ_π, qd1.π .* qd2.π)
end

function Base.:+(qd1::CrystPlastDualQD, qd2::CrystPlastDualQD)
    CrystPlastDualQD(qd1.σ + qd2.σ, qd1.ε + qd2.ε, qd1.ε_p + qd2.ε_p, qd1.τ_di + qd2.τ_di, qd1.τ + qd2.τ, qd1.γ + qd2.γ, qd1.χ⟂ + qd2.χ⟂, qd1.χo + qd2.χo,
                     qd1.ψek + qd2.ψek, qd1.ψgk + qd2.ψgk, qd1.ψe + qd2.ψe, qd1.ψg + qd2.ψg, qd1.χ_π + qd2.χ_π, qd1.ϕ_π + qd2.ϕ_π , qd1.τ_π + qd2.τ_π, qd1.π + qd2.π)
end

function create_quadrature_data{dim}(::DualProblem, ::Type{Dim{dim}}, quad_rule, nslip, n_elements)
    n_qpoints = length(points(quad_rule))
    mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:n_elements]
    temp_mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:n_elements]
    QuadratureData(mss, temp_mss)
end
