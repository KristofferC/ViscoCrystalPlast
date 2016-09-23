
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

type SerializedCrystPlastMP{dim, T, M, N}
    nslips::Int
    v::Vector{Float64}
end


function JLD.writeas{dim, T, M, N}(datam::Vector{CrystPlastMP{dim, T, M, N}})
    data = Float64[]
    for v in datam
        serialize!(data, v)
    end
    SerializedCrystPlastMP{dim, T, M, N}(length(datam[1].angles), data)
end




function JLD.readas{dim, T, M, N}(serdata::SerializedCrystPlastMP{dim, T, M, N})
    nslips = serdata.nslips
    v = serdata.v
    rank2_size = dim == 3 ? 6 : 3
    rank4_size = dim == 3 ? 36 : 9
    tot_len = 8 + rank4_size + nslips * 3 * rank2_size + nslips + nslips*nslips + 3 * nslips * dim
    @assert mod(length(v), tot_len) == 0
    n_types = length(v) ÷ tot_len
    qdata = Vector{CrystPlastMP{dim, T, M, N}}(n_types)
    c_len = 1
    for i in 1:n_types
        E, c_len = scalar(v, c_len)
        Ee = SymmetricTensor{4, dim}(v[c_len:c_len+rank4_size-1]); c_len += rank4_size
        ν, c_len = scalar(v, c_len)
        n, c_len = scalar(v, c_len)
        H⟂, c_len = scalar(v, c_len)
        Ho, c_len = scalar(v, c_len)
        Hgrad, c_len = vector(v, SymmetricTensor{2, dim, T, M}, rank2_size, nslips, c_len)
        lα, c_len = scalar(v, c_len)
        tstar, c_len = scalar(v, c_len)
        C, c_len = scalar(v, c_len)
        angles = v[c_len : c_len + nslips - 1]; c_len += nslips
        sxm_sym, c_len = vector(v, SymmetricTensor{2, dim, T, M}, rank2_size, nslips, c_len)
        Dαβ = reshape(v[c_len : c_len + nslips * nslips - 1], (nslips, nslips))
        c_len += nslips * nslips
        Esm, c_len = vector(v, SymmetricTensor{2, dim, T, M}, rank2_size, nslips, c_len)
        s, c_len = vector(v, Vec{dim, T}, dim, nslips, c_len)
        m, c_len = vector(v, Vec{dim, T}, dim, nslips, c_len)
        l, c_len = vector(v, Vec{dim, T}, dim, nslips, c_len)
        qdata[i] = CrystPlastMP(E, Ee, ν, n, H⟂, Ho, Hgrad, lα, tstar, C, angles, sxm_sym, Dαβ, Esm, s, m, l)
    end
    return qdata
end

function CrystPlastMP{T}(::Type{Dim{2}}, E::T, ν::T, n::T, H⟂::T, Ho::T, lα::T, tstar::T, C::T, angles::Vector{T})
    dim = 2
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



function CrystPlastMP{T}(::Type{Dim{3}}, E::T, ν::T, n::T, H⟂::T, Ho::T, lα::T, tstar::T, C::T, ϕs::Vector{NTuple{3, T}})
    dim = 3
    nslip = length(ϕs)
    sxm_sym = SymmetricTensor{2, dim}[]
    s = Vector{Vec{dim, Float64}}(nslip)
    m = Vector{Vec{dim, Float64}}(nslip)
    l = Vector{Vec{dim, Float64}}(nslip)

    for α = 1:nslip
        ϕ = ϕs[α]
        Q = tformrotate([0,0,1], ϕ[3]) * tformrotate([0,1,0], ϕ[2]) * tformrotate([1,0,0], ϕ[1])
        s[α] = Tensor{1, 3}(Q * [1,0,0])
        m[α] = Tensor{1, 3}(Q * [0,1,0])
        l[α] = Tensor{1, 3}(Q * [0,0,1])
    end

    λ = E*ν / ((1+ν) * (1 - 2ν))::Float64
    μ = E / (2(1+ν))::Float64
    δ(i,j) = i == j ? 1.0 : 0.0
    f(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l) * δ(j,k))
    Ee = SymmetricTensor{4, dim}(f)

    sxm_sym = [symmetric(s[α] ⊗ m[α]) for α in 1:nslip]
    Hgrad = [symmetric(H⟂ * s[α] ⊗ s[α] + Ho * l[α] ⊗ l[α]) for α in 1:nslip]
    Dαβ = Float64[(Ee ⊡ sxm_sym[α]) ⊡ sxm_sym[β] for α in 1:nslip, β in 1:nslip]
    Esm = typeof(Ee ⊡ sxm_sym[1])[Ee ⊡ sxm_sym[α] for α in 1:nslip]

    CrystPlastMP(E, Ee, ν, n, H⟂, Ho, Hgrad, lα, tstar, C, zeros(nslip), sxm_sym, Dαβ, Esm, s, m, l)
end
