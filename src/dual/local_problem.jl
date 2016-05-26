using BlockArrays
using Parameters
using ViscoCrystalPlast

using JuAFEM
using ForwardDiff
using ContMechTensors


import ViscoCrystalPlast: GeometryMesh, Dofs, DirichletBoundaryConditions, CrystPlastMP, QuadratureData, CrystPlastDualQD

@enum BlocksInner γ◫ = 1 τ◫ = 2
@enum BlocksOuter ε◫ = 1 χ⟂◫ = 2 χo◫ = 3


if !(isdefined(:DualLocalProblem))
    @eval begin
    immutable DualLocalProblem{dim, T}
        # Inner
        γ::Vector{T}
        τ::Vector{T}

        # Outer
        ε::Vector{T}
        χ⟂::Vector{T}
        χo::Vector{T}

        # Residual
        J_ττ::Matrix{T}
        J_τγ::Matrix{T}
        J_γτ::Matrix{T}
        J_γγ::Matrix{T}
        J::PseudoBlockMatrix{T, Matrix{T}}
        R_τ::Vector{T}
        R_γ::Vector{T}
        R::PseudoBlockVector{T, Vector{T}}


        # ATS tensor
        Q_γε ::Matrix{T}
        Q_γχ⟂::Matrix{T}
        Q_γχo::Matrix{T}
        Q_τε ::Matrix{T}
        Q_τχ⟂::Matrix{T}
        Q_τχo::Matrix{T}
        Q::PseudoBlockMatrix{T, Matrix{T}}

        A::BlockMatrix{T, Matrix{T}}

        outer::PseudoBlockVector{T, Vector{T}}
        inner::PseudoBlockVector{T, Vector{T}}
    end
    end
end


function DualLocalProblem{dim}(nslips::Int, ndim::Type{Dim{dim}})
    T = Float64

    if dim == 2
        ncomp = 4
    elseif dim == 3
        ncomp = 9
    else
        error("invalid dim")
    end

    γ = zeros(T, nslips)
    τ = zeros(T, nslips)
    ε = zeros(T, ncomp)
    χ⟂ = zeros(T, nslips)
    χo = zeros(T, nslips)

    J_ττ = zeros(T, nslips, nslips)
    J_τγ = zeros(T, nslips, nslips)
    J_γτ = zeros(T, nslips, nslips)
    J_γγ = zeros(T, nslips, nslips)
    J    = PseudoBlockArray(zeros(T, 2*nslips, 2*nslips), [nslips, nslips], [nslips, nslips])
    R_τ  = zeros(T, nslips)
    R_γ  = zeros(T, nslips)
    R    = PseudoBlockArray(zeros(T, 2*nslips), [nslips, nslips])
    dR    = PseudoBlockArray(zeros(T, 2*nslips), [nslips, nslips])

    Q_γε = zeros(T, nslips, ncomp)
    Q_γχ⟂ = zeros(T, nslips, nslips)
    Q_γχo = zeros(T, nslips, nslips)
    Q_τε = zeros(T, nslips, ncomp)
    Q_τχ⟂ = zeros(T, nslips, nslips)
    Q_τχo = zeros(T, nslips, nslips)

    if dim == 2
        Q = PseudoBlockArray(zeros(T, 2*nslips, ncomp+nslips), [nslips, nslips], [ncomp, nslips])
        outer = PseudoBlockArray(zeros(T, ncomp+nslips), [ncomp, nslips])
        A = BlockArray(zeros(T, 2*nslips, ncomp+2*nslips), [nslips, nslips], [ncomp, nslips, nslips])
    else
        Q = PseudoBlockArray(zeros(T, 2*nslips, ncomp+2*nslips), [nslips, nslips], [ncomp, nslips, nslips])
        outer = PseudoBlockArray(zeros(T, ncomp+2*nslips), [ncomp, nslips, nslips])
        A = BlockArray(zeros(T, 2*nslips, ncomp+2*nslips), [nslips, nslips], [ncomp, nslips, nslips])
    end
    inner = PseudoBlockArray(zeros(T, 2*nslips), [nslips, nslips])
    Δinner = PseudoBlockArray(zeros(T, 2*nslips), [nslips, nslips])
    DualLocalProblem{dim, T}(γ, τ, ε, χ⟂, χo, J_ττ, J_τγ, J_γτ, J_γγ, J, R_τ, R_γ, R,
                             Q_γε, Q_γχ⟂, Q_γχo, Q_τε, Q_τχ⟂, Q_τχo, Q, A, outer, inner)
end

function reset!(dlp::DualLocalProblem)
    fill!(full(dlp.J), 0.0)
    fill!(full(dlp.R), 0.0)
    fill!(full(dlp.Q), 0.0)
    return dlp
end

function update_problem!{dim}(problem::DualLocalProblem{dim}, Δt, mp, ms)
    @unpack problem: γ, τ, ε, χ⟂, χo, J_ττ, J_τγ, J_γτ, J_γγ, J, R_τ, R_γ, R, outer, inner

    nslip = length(mp.angles)

    @inbounds begin

    getblock!(ε, outer, ε◫)
    getblock!(χ⟂, outer, χ⟂◫)
    if dim == 3
        getblock!(χo, outer, χo◫)
    end
    εt = symmetric(Tensor{2, dim}(ε))

    getblock!(τ, inner, τ◫)
    getblock!(γ, inner, γ◫)

    ε_p = zero(εt)
    for α in 1:length(γ)
        ε_p += γ[α] * mp.sxm_sym[α]
    end

    ε_e = εt - ε_p
    σ = mp.Ee ⊡ ε_e

    for α in 1:nslip
        # Residual
        τen = -(σ ⊡ mp.sxm_sym[α])
        Δγ = γ[α] - ms.γ[α]
        R_γ[α] = τen + τ[α] - χ⟂[α]
        if dim == 3
            R_γ[α] -= χo[α]
        end
        R_τ[α] = Δγ - Δt / mp.tstar * (abs(τ[α]) / mp.C )^(mp.n) * sign(τ[α])

        # Jacobian
        for β in 1:nslip
            J_γγ[α, β] = mp.Dαβ[α, β]
            if α == β
                J_τγ[α, β] = 1.0
                J_ττ[α, β] = - Δt / mp.tstar * mp.n / mp.C * (abs(τ[α]) / mp.C )^(mp.n-1)
                J_γτ[α, β] = 1.0
            end
        end
    end

    R[γ◫] = R_γ
    R[τ◫] = R_τ

    J[γ◫, τ◫] = J_γτ
    J[γ◫, γ◫] = J_γγ
    J[τ◫, τ◫] = J_ττ
    J[τ◫, γ◫] = J_τγ

    end # inbounds

    return
end

function update_ats!{dim}(problem::DualLocalProblem{dim}, Δt, mp, ms)
    @unpack problem: γ, τ, ε, χ⟂, χo, Q_γε, Q_γχ⟂, Q_γχo, Q_τε, Q_τχ⟂, Q_τχo, Q
    nslip = length(mp.angles)

    @inbounds begin
    for α in 1:nslip
        Q_γε[α, :] = -mp.Esm[α]

        Q_γχ⟂[α, α] = -1.0
        if dim == 3
            Q_γχo[α, α] = -1.0
        end
    end

    Q[γ◫, ε◫] = Q_γε
    Q[γ◫, χ⟂◫] = Q_γχ⟂
    if dim == 3
        Q[γ◫, χo◫] = Q_γχo
    end
    end # inbounds
end

function consistent_tangent{dim}(out::Vector, problem::DualLocalProblem{dim}, ∆t, mp, ms, temp_ms)
    # The problem here is assumed to be solved and the results are in the gauss point data in temp_ms.
    reset!(problem)
    problem.inner[τ◫] = temp_ms.τ_di
    problem.inner[γ◫] = temp_ms.γ
    copy!(full(problem.outer), out)

    update_problem!(problem, ∆t, mp, ms)
    @assert norm(problem.R) <= 1e-6
    update_ats!(problem, ∆t, mp, ms)
    # copy!(problem.A,

    return -inv(full(problem.J)) * full(problem.Q)
end


function solve_local_problem{T}(out::Vector{T}, problem::DualLocalProblem, ∆t, mp, ms, temp_ms)
    # Set initial value

    problem.inner[γ◫] = ms.γ
    problem.inner[τ◫] = ms.τ_di

    copy!(full(problem.outer), out)

    max_iters = 40
    n_iters = 1

    while true
        reset!(problem)
        update_problem!(problem, ∆t, mp, ms)
        res = norm(full(problem.R), Inf)

        if res  <= 1e-7
            break
        end

        A_ldiv_B!(lufact!(full(problem.J)), full(problem.R))
        @inbounds for i in eachindex(full(problem.inner))
            problem.inner[i] -= problem.R[i]
        end
        if n_iters == max_iters
            error("Non conv mat")
        end
        n_iters +=1
    end

    return full(problem.inner)
end

#=
function setup_material{dim}(::Type{Dim{dim}}, lα)
    E = 200000.0
    ν = 0.3
    n = 2.0
    #lα = 0.5
    H⟂ = 0.1E
    Ho = 0.1E
    C = 1.0e3
    tstar = 1000.0
    angles = [20.0, 40.0]
    mp = ViscoCrystalPlast.CrystPlastMP(Dim{dim}, E, ν, n, H⟂, Ho, lα, tstar, C, angles)
    return mp
end




function foo()
    srand(1234)
    nslip = 2
    dim = 2
    a = rand(dim == 2? 4 : 9);
    b = rand(nslip);
    c = rand(nslip);
    d =  rand(nslip);

    problem = DualLocalProblem(nslip, Dim{dim});
    out = [a; b] #rand(dim == 2? 4 : 9);
    problem.inner[γ◫] = c; #rand(nslip);
    problem.inner[τ◫] = d; #rand(nslip);

    mp = setup_material(Dim{dim}, 0.1);
    ms = CrystPlastDualQD(nslip, Dim{dim});
    temp_ms = CrystPlastDualQD(nslip, Dim{dim});


    X = solve_local_problem(out, problem, 0.1, mp, ms, temp_ms)
    #copy!(full(problem.inner), X)
    temp_ms.γ = problem.inner[γ◫]
    temp_ms.τ_di = problem.inner[τ◫]
    @time for i in 1:10^5
        consistent_tangent(out, problem, 0.1, mp, ms, temp_ms)
    end


    @time for i in 1:10^5
        update_problem!(problem, 0.1, mp, ms)
        consistent_tangent(out, 0.1, mp, temp_ms, problem)
    end


end
=#