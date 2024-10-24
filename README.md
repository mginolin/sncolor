This repository allows to fit and plot a SN colour distribution following the intrinsic + dust model (Jha el al. 2007, Mandel et al. 2011, Brout & Scolnic 2021, Ginolin et al. 2024b).
The model is following:

$$  P(c) = \mathcal{N}(c|c_\mathrm{int}, \sigma_c) \otimes
\begin{cases}
0  & \text{if $c \leq 0$} \\
\frac{1}{\tau}e^{-c/\tau} & \text{if $c>0$}
\end{cases} $$

### Fitting for the full colour distribution
You need a `df` pandas dataframe with columns `c` and `c_err`.

    #Initialising the colour object
    a = sncolor.ColorFit.from_dataset(df)
    a.set_dust_only(False)
    a.full_fit()
    print(a.mu_int, a.mu_int_err)
    print(a.sig_int, a.sig_int_err)
    print(a.tau_dust, a.tau_dust_err)
You can also access the full iminuit object with `a.minuit`.

### Plotting the colour distribution
You can plot either the full distribution, or only the intrinsic or dust part.

    #The plotting range has to be symmetric for the distributions to be centered correctly
    xc = np.linspace(-1, 1, 2000)
    plt.plot(xc, sncolor.fit_function_dust(xc, 0.1), label='Dust extinction', lw=1, linestyle='dashed', color='tab:blue')
    plt.plot(xc, sncolor.fit_function_sne(xc, -0.1, 0.05), label='Intrinsic color', lw=1, linestyle='dotted', color='tab:blue')
    plt.plot(xc, sncolor.fit_function_tot(xc, 0.1, -0.1, 0.05), label='Full fit', lw=3, color='tab:orange')
    plt.xlabel('Color $c$', fontsize='x-large')
    plt.show()

