using JLD
using DataFrames
using DataFramesMeta
using Plots

plotly()
function analyze_error()

        primal_ok_slips = [
    0.062191661532416447,
    0.0571484797877182,
    0.0517745718362789,
    0.046403569755167816,
    0.04126535181951812,
    0.036507986411229404,
    0.03221521441349259,
    0.0284151209072776,
    0.02509463859631519,
    0.022216148033211307]



    primal_ok_grad_en = [
    3.6514653912685073,
    5.181457188670423,
    6.2689332503865725,
    7.04725115903481,
    7.562299024830206,
    7.855557596734455,
    7.972850204800356,
    7.958992463836155,
    7.852768881624368,
    7.684979033673338]


    df_dual = load("dataframes/dual_data_frame.jld")["df"]
    df_primal = load("dataframes/primal_data_frame.jld")["df"]

#    p = plot()
    p2 = plot()
    for (i, l) in enumerate(unique(df_dual[:l]))
        df_dual_l = @where(df_dual, :l .== l)
        df_primal_l = @where(df_primal, :l .== l)

        eles = df_dual_l[:n_elements]
        #dual_err_grad_relative = df_dual_l[:err_grad_energy] ./ primal_ok_grad_slips[i]
        primal_err_grad_relative = df_primal_l[:err_grad_energy] ./ primal_ok_grad_en[i]

      #  plot!(p, eles, dual_err_grad_relative, label ="$l")
        plot!(p2, eles, primal_err_grad_relative, label="hej", legend=true)
    end
end

analyze_error()

pgfplots()
 using LaTeXStrings
function analyze_grad_energy_with_l()
    df = load("dataframes/primal_l_study.jld")["df"]
    ps = plot(df[:l], df[:tot_slip],
            xlabel = L"$\overline{\gamma}$",
            ylabel = L"$l$",
            title = L"Effective slip $\overline{\gamma} = \sqrt{\int \sum_{\alpha} \gamma_{\alpha}^2 dV}$"
    )
    pg = plot(df[:l], df[:tot_grad_energy],
        xlabel = L"$\int\Psi_g$",
        ylabel = L"$l$",
        title = "Gradient energy"
    )
    return ps, pg
end

ps, pg = analyze_grad_energy_with_l()
ps
pg
PGFPlots.save("plots/slip.tex", ps.o)
PGFPlots.save("plots/grad_en.tex", pg.o)

using Plots
pgfplots()
function g()
    p = plot(rand(5), rand(5))
    return p
end

p = g()