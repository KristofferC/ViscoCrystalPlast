type CrystPlastPrimalQD{dim, T, M}
    σ::SymmetricTensor{2, dim, T, M}
    ε::SymmetricTensor{2, dim, T, M}
    ε_p::SymmetricTensor{2, dim, T, M}
    τ_di::Vector{T}
    τ::Vector{T}
    g::Vector{Vec{dim, T}}
end

function CrystPlastPrimalQD{dim}(nslip, ::Type{Dim{dim}})
    σ = zero(SymmetricTensor{2, dim})
    ε = zero(SymmetricTensor{2, dim})
    ε_p = zero(SymmetricTensor{2, dim})
    τ_di = zeros(nslip)
    τ = zeros(nslip)
    g = Vec{dim, Float64}[zero(Vec{dim, Float64}) for i in 1:nslip]
    return CrystPlastPrimalQD(σ, ε, ε_p, τ_di, τ, g)
end
