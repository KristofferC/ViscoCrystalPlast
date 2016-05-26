immutable CrystPlastMP{dim, T, M, N}
    E::T
    Ee::SymmetricTensor{4, dim, T, N}
    ν::T
    n::T
    H⟂::T
    Ho::T
    Hgrad::Vector{SymmetricTensor{2, dim, T, M}}
    lα::T
    tstar::T
    C::T
    angles::Vector{T}
    sxm_sym::Vector{SymmetricTensor{2, dim, T, M}}
    Dαβ::Matrix{T}
    Esm::Vector{SymmetricTensor{2, dim, T, M}}
    s::Vector{Vec{dim, T}}
    m::Vector{Vec{dim, T}}
    l::Vector{Vec{dim, T}}
end


function CrystPlastMP{T, dim}(::Type{Dim{dim}}, E::T, ν::T, n::T, H⟂::T, Ho::T, lα::T, tstar::T, C::T, angles::Vector{T})
    nslip = length(angles)
    sxm_sym = SymmetricTensor{2, dim}[]
    s = Vector{Vec{dim, T}}(nslip)
    m = Vector{Vec{dim, T}}(nslip)
    l = Vector{Vec{dim, T}}(nslip)

    λ = E*ν / ((1+ν) * (1 - 2ν))::Float64
    μ = E / (2(1+ν))::Float64
    δ(i,j) = i == j ? 1.0 : 0.0
    f(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l) * δ(j,k))
    Ee = SymmetricTensor{4, dim}(f)

    for α = 1:nslip
        t = deg2rad(angles[α])
        s_a = Tensor{1, 3}([cos(t), sin(t), 0.0])
        m_a = Tensor{1, 3}([cos(t + pi/2), sin(t + pi/2), 0.0])
        s[α] = convert(Tensor{1, dim}, s_a)
        m[α] = convert(Tensor{1, dim}, m_a)
        l[α] = convert(Tensor{1, dim}, Tensor{1, 3}(cross(s_a, m_a)))
    end

    sxm_sym = [symmetric(s[α] ⊗ m[α]) for α in 1:nslip]
    Hgrad = [symmetric(H⟂ * s[α] ⊗ s[α] + Ho * l[α] ⊗ l[α]) for α in 1:nslip]
    Dαβ = T[(Ee ⊡ sxm_sym[α]) ⊡ sxm_sym[β] for α in 1:nslip, β in 1:nslip]
    Esm = typeof(Ee ⊡ sxm_sym[1])[Ee ⊡ sxm_sym[α] for α in 1:nslip]

    CrystPlastMP(E, Ee, ν, n, H⟂, Ho, Hgrad, lα, tstar, C, angles, sxm_sym, Dαβ, Esm, s, m, l)
end
