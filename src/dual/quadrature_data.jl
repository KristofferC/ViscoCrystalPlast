type CrystPlastDualQD{dim, T, M}
    σ::SymmetricTensor{2, dim, T, M}
    ε::SymmetricTensor{2, dim, T, M}
    ε_p::SymmetricTensor{2, dim, T, M}
    τ_di::Vector{T}
    τ::Vector{T}
    γ::Vector{T}
    χ::Vector{T}
end

function CrystPlastDualQD{dim}(nslip, ::Type{Dim{dim}})
    σ = zero(SymmetricTensor{2, dim})
    ε = zero(SymmetricTensor{2, dim})
    ε_p = zero(SymmetricTensor{2, dim})
    τ_di = zeros(nslip)
    τ = zeros(nslip)
    γ = zeros(nslip)
    χ = zeros(nslip)
    return CrystPlastDualQD(σ, ε, ε_p, τ_di, τ, γ, χ)
end

function create_quadrature_data{dim}(::DualProblem, ::Type{Dim{dim}}, quad_rule, nslip, n_elements)
    n_qpoints = length(points(quad_rule))
    mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:n_elements]
    temp_mss = [CrystPlastPrimalQD(nslip, Dim{dim}) for i = 1:n_qpoints, j = 1:n_elements]
    QuadratureData(mss, temp_mss)
end
