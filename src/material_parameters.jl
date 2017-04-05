
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

function CrystPlastMP{dim, T}(::Type{Dim{dim}}, E::T, ν::T, n::T, H⟂::T, Ho::T, lα::T, tstar::T, C::T, s::Vector, m::Vector, l::Vector)
    nslip = length(s)
    @assert length(s) == length(m) == length(l)
    sxm_sym = SymmetricTensor{2, dim}[]

    λ = E*ν / ((1+ν) * (1 - 2ν))::Float64
    μ = E / (2(1+ν))::Float64
    δ(i,j) = i == j ? 1.0 : 0.0
    f(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l) * δ(j,k))
    Ee = SymmetricTensor{4, dim}(f)

    sxm_sym = [symmetric(s[α] ⊗ m[α]) for α in 1:nslip]
    Hgrad = [symmetric(H⟂ * s[α] ⊗ s[α] + Ho * l[α] ⊗ l[α]) for α in 1:nslip]
    Dαβ = T[(Ee ⊡ sxm_sym[α]) ⊡ sxm_sym[β] for α in 1:nslip, β in 1:nslip]
    Esm = typeof(Ee ⊡ sxm_sym[1])[Ee ⊡ sxm_sym[α] for α in 1:nslip]
    angles = zeros(nslip)
    CrystPlastMP(E, Ee, ν, n, H⟂, Ho, Hgrad, lα, tstar, C, angles, sxm_sym, Dαβ, Esm, s, m, l)
end

function CrystPlastMP{T}(::Type{Dim{2}}, E::T, ν::T, n::T, H⟂::T, Ho::T, lα::T, tstar::T, C::T, angles::Vector{T})
    dim = 2
    for α = 1:nslip
        t = deg2rad(angles[α])
        s_a = Tensor{1, 3}([cos(t), sin(t), 0.0])
        m_a = Tensor{1, 3}([cos(t + pi/2), sin(t + pi/2), 0.0])
        s[α] = dim == 3 ? s_a : Vec{2}((s_a[1], s_a[2]))
        m[α] = dim == 3 ? m_a : Vec{2}((m_a[1], m_a[2]))
        l[α] = zero(Vec{dim}) # convert(Tensor{1, dim}, cross(s_a, m_a))
    end

    sxm_sym = [symmetric(s[α] ⊗ m[α]) for α in 1:nslip]
    Hgrad = [symmetric(H⟂ * s[α] ⊗ s[α] + Ho * l[α] ⊗ l[α]) for α in 1:nslip]
    Dαβ = T[(Ee ⊡ sxm_sym[α]) ⊡ sxm_sym[β] for α in 1:nslip, β in 1:nslip]
    Esm = typeof(Ee ⊡ sxm_sym[1])[Ee ⊡ sxm_sym[α] for α in 1:nslip]

    CrystPlastMP(E, Ee, ν, n, H⟂, Ho, Hgrad, lα, tstar, C, angles, sxm_sym, Dαβ, Esm, s, m, l)
end


function CrystPlastMP{T}(::Type{Dim{3}}, E::T, ν::T, n::T, H⟂::T, Ho::T, lα::T, tstar::T, C::T, s::Vector, m::Vector)
    nslip = length(s)
    @assert length(s) == length(m)

    l = Vector{Vec{3, Float64}}(nslip)

    for α = 1:nslip
        l[α] = m[α] × s[α]
    end

    CrystPlastMP(Dim{3}, E, ν, n, H⟂, Ho, lα, tstar, C, s, m, l)
end



function CrystPlastMP{T}(::Type{Dim{3}}, E::T, ν::T, n::T, H⟂::T, Ho::T, lα::T, tstar::T, C::T, rotmats::Vector)
    dim = 3
    nslip = length(rotmats)
    sxm_sym = SymmetricTensor{2, dim}[]
    s = Vector{Vec{3, Float64}}(nslip)
    m = Vector{Vec{3, Float64}}(nslip)
    l = Vector{Vec{3, Float64}}(nslip)

    for α = 1:nslip
        s[α] = Tensor{1, 3}(rotmats[α] * [1,0,0])
        m[α] = Tensor{1, 3}(rotmats[α] * [0,1,0])
        l[α] = Tensor{1, 3}(rotmats[α] * [0,0,1])
    end

    CrystPlastMP(Dim{3}, E, ν, n, H⟂, Ho, lα, tstar, C, s, m, l)
end
