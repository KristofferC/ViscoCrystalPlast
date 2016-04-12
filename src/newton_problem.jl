immutable NewtonProblem
    K::SparseMatrixCSC{Float64, Int}
    f_full::Vector{Float64}
    f_cond::Vector{Float64}
    x::Vector{Float64}
    prev_x::Vector{Float64}
    trial_x::Vector{Float64}
    ∆x::Vector{Float64}
end

function NewtonProblem(K::SparseMatrixCSC, n_full, n_free)
    ∆x = 0.0001 * ones(n_free)
    x = zeros(n_full)
    prev_x = copy(x)
    trial_x = copy(x)
    f_full = copy(x)
    f_cond = zeros(∆x)
    NewtonProblem(K, f_full, f_cond, x, prev_x, trial_x, ∆x)
end

function solve(np::NewtonProblem, g, free)
    iter = 1
    n_iters = 20

    while iter == 1 || norm(f, Inf)  >= 1e-6
        np.trial_x[free] = np[free] + np.∆x
        @timer "total K" K_condensed, f = g!(newton_problem)

        @timer "factorization" np.∆x -=  K[free, free] \ f[free]
        println(norm(f))
        iter +=1
        if n_iters == iter
            error("too many iterations without convergence")
        end
    end
end