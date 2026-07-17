# This file is included from `OpenSpins_impl.jl` and extends module `OpenSpins`.

export spin_expectation_from_raw
export two_spin_correlator_from_raw
export purity_from_raw
export schwinger_boson_constraint_from_raw

"""
Compute <s_n^alpha>(t) from a serialized raw-output payload.
"""
function spin_expectation_from_raw(raw)
    gK = raw.gK
    return spin_expectation_from_gk(gK)
end

"""
Connected two-spin correlator from Eq. <s_n^alpha(t) s_{n'}^beta(t')> = i/4 * Mcheck.
"""
function two_spin_correlator_from_raw(raw, n::Integer, np::Integer, alpha::Integer, beta::Integer; t::Integer, tp::Integer)
    a = component_index(Int(alpha))
    b = component_index(Int(beta))
    return 0.25im * raw.mcheckK[Int(n), Int(np), a, b, t, tp]
end

"""
Purity proxy used in the manuscript for S=1/2: |P| = 2 * sqrt(sum_alpha <s_alpha>^2).
"""
function purity_from_raw(raw)
    s = spin_expectation_from_raw(raw)
    n_spins = size(s, 1)
    nt = size(s, 3)
    out = zeros(Float64, n_spins, nt)
    for n in 1:n_spins, t in 1:nt
        out[n, t] = 2 * sqrt(sum(s[n, alpha, t]^2 for alpha in 1:N_COMP))
    end
    return out
end

"""
Compute the Schwinger-boson occupation constraint
`<a^dagger a + b^dagger b>(t)` from the equal-time Keldysh Green function.
For a spin of length `S`, the constrained value is `2S`.
"""
function schwinger_boson_constraint_from_raw(raw)
    gK = raw.gK
    n_spins = size(gK, 1)
    nt = size(gK, 4)
    out = zeros(Float64, n_spins, nt)
    for n in 1:n_spins, t in 1:nt
        out[n, t] = real(0.25im * tr(@view(gK[n, :, :, t, t]))) - 1
    end
    return out
end
