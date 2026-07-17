
using KadanoffBaym
using LinearAlgebra
using Integrals
using Interpolations
using Serialization
using Printf

export BathSpectrum
export InteractionProfile
export Quench
export LinearRamp
export SmoothRamp
export OpenSpinsParameters
export OpenSpinsInitialState
export run_simulation
export save_raw_output
export load_raw_output
export audit_solution
export spin_expectation_from_gk
export component_index

const N_FIELD = 4
const N_COMP = 3
const COMP_LABELS = (1, 2, 3)

const SIGMA_X = ComplexF64[0 1; 1 0]
const SIGMA_Y = ComplexF64[0 -im; im 0]
const SIGMA_Z = ComplexF64[1 0; 0 -1]
const I2 = Matrix{ComplexF64}(I, 2, 2)

const K0_MATRIX = kron(I2, SIGMA_Y)
const K_MATRICES = (
    kron(SIGMA_X, I2),
    -kron(SIGMA_Y, SIGMA_Y),
    kron(SIGMA_Z, I2),
)

struct TraceStencilTerm
    coeff::ComplexF64
    i::Int
    j::Int
    k::Int
    l::Int
end

function _build_trace_stencils()
    stencils = Matrix{Vector{TraceStencilTerm}}(undef, N_COMP, N_COMP)
    for alpha in 1:N_COMP, beta in 1:N_COMP
        Ka = K_MATRICES[alpha]
        Kb = K_MATRICES[beta]
        terms = TraceStencilTerm[]
        for i in 1:N_FIELD, j in 1:N_FIELD, k in 1:N_FIELD, l in 1:N_FIELD
            coeff = Ka[i, j] * Kb[k, l]
            iszero(coeff) && continue
            push!(terms, TraceStencilTerm(coeff, i, j, k, l))
        end
        stencils[alpha, beta] = terms
    end
    return stencils
end

const TRACE_STENCILS = _build_trace_stencils()

"""
Map spin-component labels to integer index.
"""
function component_index(component::Integer)
    idx = Int(component)
    1 <= idx <= N_COMP || error("Unknown spin component index: $idx. Expected 1, 2, or 3.")
    return idx
end

# -----------------------------------------------------------------------------
# Custom symmetries
# -----------------------------------------------------------------------------

"""Symmetry for g_n^{ab}(t,t') and Sigma_n^{ab}(t,t'): transpose only a,b."""
struct SymmetricSiteField <: KadanoffBaym.AbstractSymmetry end
@inline KadanoffBaym.symmetry(::Type{SymmetricSiteField}) = v -> permutedims(v, (1, 3, 2))

"""Antisymmetry for g_n^s(t,t') and Sigma_n^s(t,t'): minus-transpose on a,b."""
struct AntiSymmetricSiteField <: KadanoffBaym.AbstractSymmetry end
@inline KadanoffBaym.symmetry(::Type{AntiSymmetricSiteField}) = v -> -permutedims(v, (1, 3, 2))

"""Symmetry for M^{alpha,beta}_{n,n'}: transpose in combined (n,alpha) matrix indices."""
struct SymmetricMeanFieldTensor <: KadanoffBaym.AbstractSymmetry end
@inline KadanoffBaym.symmetry(::Type{SymmetricMeanFieldTensor}) = v -> permutedims(v, (2, 1, 4, 3))

"""Antisymmetry for M^s in combined (n,alpha) matrix indices."""
struct AntiSymmetricMeanFieldTensor <: KadanoffBaym.AbstractSymmetry end
@inline KadanoffBaym.symmetry(::Type{AntiSymmetricMeanFieldTensor}) = v -> -permutedims(v, (2, 1, 4, 3))

"""Keldysh-like symmetry for scalar channels (label axes are not transposed)."""
struct SymmetricScalarChannel <: KadanoffBaym.AbstractSymmetry end
@inline KadanoffBaym.symmetry(::Type{SymmetricScalarChannel}) = identity

"""Spectral-like antisymmetry for scalar channels (label axes are not transposed)."""
struct AntiSymmetricScalarChannel <: KadanoffBaym.AbstractSymmetry end
@inline KadanoffBaym.symmetry(::Type{AntiSymmetricScalarChannel}) = -

# Avoid materializing permutedims/transmute results when mirroring time points.
Base.@propagate_inbounds function Base.setindex!(G::GreenFunction{T,5,A,SymmetricSiteField}, v, i1::Int, i2::Int) where {T,A}
    data = G.data
    @inbounds for n in axes(data, 1), a in axes(data, 2), b in axes(data, 3)
        val = v[n, a, b]
        data[n, a, b, i1, i2] = val
        if i1 != i2
            data[n, b, a, i2, i1] = val
        end
    end
    return v
end

Base.@propagate_inbounds function Base.setindex!(G::GreenFunction{T,5,A,AntiSymmetricSiteField}, v, i1::Int, i2::Int) where {T,A}
    data = G.data
    @inbounds for n in axes(data, 1), a in axes(data, 2), b in axes(data, 3)
        val = v[n, a, b]
        data[n, a, b, i1, i2] = val
        if i1 != i2
            data[n, b, a, i2, i1] = -val
        end
    end
    return v
end

Base.@propagate_inbounds function Base.setindex!(G::GreenFunction{T,6,A,SymmetricMeanFieldTensor}, v, i1::Int, i2::Int) where {T,A}
    data = G.data
    @inbounds for n in axes(data, 1), np in axes(data, 2), alpha in axes(data, 3), beta in axes(data, 4)
        val = v[n, np, alpha, beta]
        data[n, np, alpha, beta, i1, i2] = val
        if i1 != i2
            data[np, n, beta, alpha, i2, i1] = val
        end
    end
    return v
end

Base.@propagate_inbounds function Base.setindex!(G::GreenFunction{T,6,A,AntiSymmetricMeanFieldTensor}, v, i1::Int, i2::Int) where {T,A}
    data = G.data
    @inbounds for n in axes(data, 1), np in axes(data, 2), alpha in axes(data, 3), beta in axes(data, 4)
        val = v[n, np, alpha, beta]
        data[n, np, alpha, beta, i1, i2] = val
        if i1 != i2
            data[np, n, beta, alpha, i2, i1] = -val
        end
    end
    return v
end

Base.@propagate_inbounds function Base.setindex!(G::GreenFunction{T,4,A,SymmetricScalarChannel}, v, i1::Int, i2::Int) where {T,A}
    data = G.data
    @inbounds for d in axes(data, 1), alpha in axes(data, 2)
        val = v[d, alpha]
        data[d, alpha, i1, i2] = val
        if i1 != i2
            data[d, alpha, i2, i1] = val
        end
    end
    return v
end

Base.@propagate_inbounds function Base.setindex!(G::GreenFunction{T,4,A,AntiSymmetricScalarChannel}, v, i1::Int, i2::Int) where {T,A}
    data = G.data
    @inbounds for d in axes(data, 1), alpha in axes(data, 2)
        val = v[d, alpha]
        data[d, alpha, i1, i2] = val
        if i1 != i2
            data[d, alpha, i2, i1] = -val
        end
    end
    return v
end

# -----------------------------------------------------------------------------
# User-facing model specification
# -----------------------------------------------------------------------------

"""
Bath spectral specification for one independent bosonic channel.

Fields:
- `spin`: physical spin index n in 1:N_S
- `component`: spin component alpha in {1,2,3}
- `gamma`: coupling prefactor in Eq. (J(omega))
- `s`: Ohmic exponent
- `omega_c`: cutoff frequency
- `temperature`: bath temperature (same units as frequencies)
- `omega_max`: signed-frequency integration bound uses [-omega_max, omega_max]
- `n_omega`: auxiliary metadata for reporting; Integrals.jl chooses actual quadrature nodes
"""
struct BathSpectrum
    spin::Int
    component::Int
    gamma::Float64
    s::Float64
    omega_c::Float64
    temperature::Float64
    omega_max::Float64
    n_omega::Int
end

abstract type AbstractInteractionEvent end

"""Instantaneously set an interaction profile to `target` at `time`."""
struct Quench <: AbstractInteractionEvent
    time::Float64
    target::Float64
    function Quench(time::Real, target::Real)
        isfinite(time) && time >= 0 || error("Quench time must be finite and non-negative.")
        isfinite(target) || error("Quench target must be finite.")
        new(Float64(time), Float64(target))
    end
end

"""Linearly ramp from the currently held value to `target` over `[start, stop]`."""
struct LinearRamp <: AbstractInteractionEvent
    start::Float64
    stop::Float64
    target::Float64
    function LinearRamp(start::Real, stop::Real, target::Real)
        isfinite(start) && start >= 0 || error("Ramp start must be finite and non-negative.")
        isfinite(stop) && stop > start || error("Ramp stop must be finite and greater than start.")
        isfinite(target) || error("Ramp target must be finite.")
        new(Float64(start), Float64(stop), Float64(target))
    end
end

"""Smoothstep ramp from the currently held value to `target` over `[start, stop]`."""
struct SmoothRamp <: AbstractInteractionEvent
    start::Float64
    stop::Float64
    target::Float64
    function SmoothRamp(start::Real, stop::Real, target::Real)
        isfinite(start) && start >= 0 || error("Ramp start must be finite and non-negative.")
        isfinite(stop) && stop > start || error("Ramp stop must be finite and greater than start.")
        isfinite(target) || error("Ramp target must be finite.")
        new(Float64(start), Float64(stop), Float64(target))
    end
end

@inline _event_start(event::Quench) = event.time
@inline _event_start(event::Union{LinearRamp, SmoothRamp}) = event.start
@inline _event_stop(event::Quench) = event.time
@inline _event_stop(event::Union{LinearRamp, SmoothRamp}) = event.stop

"""
Piecewise interaction scale with an initial value and ordered quench/ramp events.

Quenches are right-continuous. A ramp starts from the value held at its `start`
and holds its target after `stop`.
"""
struct InteractionProfile
    initial::Float64
    events::Vector{AbstractInteractionEvent}
end

function InteractionProfile(initial::Real = 1.0; events::AbstractVector = AbstractInteractionEvent[])
    isfinite(initial) || error("Interaction profile initial value must be finite.")
    normalized = AbstractInteractionEvent[]
    previous_stop = -Inf
    for event in events
        event isa AbstractInteractionEvent || error("Unknown interaction profile event $(typeof(event)).")
        start = _event_start(event)
        start >= previous_stop || error("Interaction profile events must be ordered and non-overlapping.")
        push!(normalized, event)
        previous_stop = _event_stop(event)
    end
    return InteractionProfile(Float64(initial), normalized)
end

@inline function (profile::InteractionProfile)(time::Real)
    t = Float64(time)
    value = profile.initial
    for event in profile.events
        if event isa Quench
            t < event.time && return value
            value = event.target
        else
            t <= event.start && return value
            if t < event.stop
                x = (t - event.start) / (event.stop - event.start)
                weight = event isa SmoothRamp ? x * x * (3 - 2x) : x
                return muladd(weight, event.target - value, value)
            end
            value = event.target
        end
    end
    return value
end

@inline _event_data(event::Quench) = (kind = :quench, start = event.time, stop = event.time, target = event.target)
@inline _event_data(event::LinearRamp) = (kind = :linear_ramp, start = event.start, stop = event.stop, target = event.target)
@inline _event_data(event::SmoothRamp) = (kind = :smooth_ramp, start = event.start, stop = event.stop, target = event.target)

interaction_profile_data(profile::InteractionProfile) = (
    initial = profile.initial,
    events = [_event_data(event) for event in profile.events],
)

function InteractionProfile(data::NamedTuple)
    events = AbstractInteractionEvent[]
    for event in data.events
        if event.kind === :quench
            push!(events, Quench(event.start, event.target))
        elseif event.kind === :linear_ramp
            push!(events, LinearRamp(event.start, event.stop, event.target))
        elseif event.kind === :smooth_ramp
            push!(events, SmoothRamp(event.start, event.stop, event.target))
        else
            error("Unknown saved interaction profile event kind $(event.kind).")
        end
    end
    return InteractionProfile(data.initial; events = events)
end

function BathSpectrum(
    spin::Integer,
    component::Integer;
    gamma::Real,
    s::Real,
    omega_c::Real,
    temperature::Real,
    omega_max::Real,
    n_omega::Integer = 800,
)
    comp = component_index(component)
    spin >= 1 || error("Bath spin index must be >= 1.")
    gamma >= 0 || error("Bath gamma must be non-negative.")
    omega_c > 0 || error("omega_c must be > 0.")
    omega_max > 0 || error("omega_max must be > 0.")
    n_omega >= 16 || error("n_omega should be at least 16 for stable kernels.")
    return BathSpectrum(Int(spin), comp, Float64(gamma), Float64(s), Float64(omega_c), Float64(temperature), Float64(omega_max), Int(n_omega))
end

"""
Global simulation parameters.

Tensor conventions (time last):
- g_n^{ab}: data[n, a, b, t, t']
- Sigma_n^{ab}: data[n, a, b, t, t']
- Pi_n^alpha, D_n^alpha: data[diss_site, alpha, t, t']
- Omega_n^{alpha,beta}: data[n, alpha, beta, t, t']
- Mcheck_{n,n'}^{alpha,beta}: data[n, n', alpha, beta, t, t']
"""
struct OpenSpinsParameters
    n_spins::Int
    h::Matrix{Float64}                # (n, alpha)
    J::Array{Float64, 4}              # (n, n', alpha, beta)
    dissipative_spins::Vector{Int}    # subset of 1:n_spins
    baths::Vector{BathSpectrum}
    spin_spin_profile::InteractionProfile
    spin_bath_profile::InteractionProfile
    tmax::Float64
    dtini::Float64
    dtmax::Float64
    atol::Float64
    rtol::Float64
    kernel_ntau::Int
    kernel_abstol::Float64
    kernel_reltol::Float64
    symmetry_tol::Float64
    output_basename::String
end

function OpenSpinsParameters(; 
    n_spins::Integer,
    tmax::Real,
    h::AbstractMatrix{<:Real} = zeros(Float64, Int(n_spins), N_COMP),
    J::AbstractArray{<:Real, 4} = zeros(Float64, Int(n_spins), Int(n_spins), N_COMP, N_COMP),
    dissipative_spins::AbstractVector{<:Integer} = Int[],
    baths::AbstractVector{BathSpectrum} = BathSpectrum[],
    spin_spin_profile::InteractionProfile = InteractionProfile(),
    spin_bath_profile::InteractionProfile = InteractionProfile(),
    dtini::Real = 1e-3,
    dtmax::Real = Inf,
    atol::Real = 1e-5,
    rtol::Real = 1e-5,
    kernel_ntau::Integer = 801,
    kernel_abstol::Real = 1e-9,
    kernel_reltol::Real = 1e-7,
    symmetry_tol::Real = 1e-7,
    output_basename::AbstractString = "openspins_raw.jls",
)
    n = Int(n_spins)
    n >= 1 || error("n_spins must be >= 1.")
    tmax > 0 || error("tmax must be > 0.")
    dtini > 0 || error("dtini must be > 0.")
    dtmax > 0 || error("dtmax must be > 0.")
    kernel_ntau >= 33 || error("kernel_ntau must be at least 33.")

    hmat = Matrix{Float64}(h)
    size(hmat) == (n, N_COMP) || error("h must have shape (n_spins, 3).")

    Jten = Array{Float64, 4}(J)
    size(Jten) == (n, n, N_COMP, N_COMP) || error("J must have shape (n_spins, n_spins, 3, 3).")

    diss = sort(unique(Int.(dissipative_spins)))
    all(1 .<= diss .<= n) || error("dissipative_spins must be subset of 1:n_spins.")

    for b in baths
        b.spin <= n || error("Bath spin index $(b.spin) exceeds n_spins=$(n).")
        in(b.spin, diss) || error("Bath at spin $(b.spin) is invalid: spin is not in dissipative_spins.")
    end

    return OpenSpinsParameters(
        n,
        hmat,
        Jten,
        diss,
        collect(baths),
        spin_spin_profile,
        spin_bath_profile,
        Float64(tmax),
        Float64(dtini),
        Float64(dtmax),
        Float64(atol),
        Float64(rtol),
        Int(kernel_ntau),
        Float64(kernel_abstol),
        Float64(kernel_reltol),
        Float64(symmetry_tol),
        String(output_basename),
    )
end

"""
Initial-state specification via spin expectation values.
"""
struct OpenSpinsInitialState
    spin_expectation::Matrix{Float64}  # (n, alpha)
    spin_length::Float64
end

function OpenSpinsInitialState(n_spins::Integer; spin_length::Real = 0.5)
    return OpenSpinsInitialState(zeros(Float64, Int(n_spins), N_COMP), Float64(spin_length))
end

function OpenSpinsInitialState(spin_expectation::AbstractMatrix{<:Real}; spin_length::Real = 0.5)
    return OpenSpinsInitialState(Matrix{Float64}(spin_expectation), Float64(spin_length))
end

# -----------------------------------------------------------------------------
# Kernel precomputation (Integrals.jl + Interpolations.jl)
# -----------------------------------------------------------------------------

struct KernelBundle{I}
    tau_grid::Vector{Float64}
    tau_nonneg::Vector{Float64}
    xiK_table::Array{ComplexF64, 3}      # (diss_site, alpha, tau_idx)
    xis_table::Array{ComplexF64, 3}       # (diss_site, alpha, tau_idx)
    xiK_interp::Matrix{I}
    xis_interp::Matrix{I}
    spin_to_diss::Vector{Int}
    diss_to_spin::Vector{Int}
    has_channel::BitMatrix                # (diss_site, alpha)
end

mutable struct OpenSpinsScratch
    dgK::Array{ComplexF64, 3}
    dgs::Array{ComplexF64, 3}
    collK_1::Matrix{ComplexF64}
    collK_2::Matrix{ComplexF64}
    colls::Matrix{ComplexF64}
    endpoint_op::Matrix{ComplexF64}
    lhs_s::Matrix{ComplexF64}
    lhs_k::Matrix{ComplexF64}
    rhs::Vector{ComplexF64}
    sol::Vector{ComplexF64}
    jmjK::Matrix{ComplexF64}
    jmjs::Matrix{ComplexF64}
end

mutable struct McheckEndpointCache
    row::Int
    scale_s::Float64
    scale_k::Float64
    fac_s::Any
    fac_k::Any
end

McheckEndpointCache() = McheckEndpointCache(0, NaN, NaN, nothing, nothing)

function OpenSpinsScratch(n_spins::Int)
    nvec = n_spins * N_COMP
    return OpenSpinsScratch(
        zeros(ComplexF64, n_spins, N_FIELD, N_FIELD),
        zeros(ComplexF64, n_spins, N_FIELD, N_FIELD),
        zeros(ComplexF64, N_FIELD, N_FIELD),
        zeros(ComplexF64, N_FIELD, N_FIELD),
        zeros(ComplexF64, N_FIELD, N_FIELD),
        zeros(ComplexF64, nvec, nvec),
        zeros(ComplexF64, nvec, nvec),
        zeros(ComplexF64, nvec, nvec),
        zeros(ComplexF64, nvec),
        zeros(ComplexF64, nvec),
        zeros(ComplexF64, N_COMP, N_COMP),
        zeros(ComplexF64, N_COMP, N_COMP),
    )
end

_scratch_count() = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads()

_make_thread_scratch(n_spins::Int) = [OpenSpinsScratch(n_spins) for _ in 1:_scratch_count()]

@inline _scratch(state) = state.scratch[Threads.threadid()]

function _invalidate_mcheck_endpoint_cache!(state)
    cache = state.mcheck_endpoint_cache
    cache.row = 0
    cache.scale_s = NaN
    cache.scale_k = NaN
    cache.fac_s = nothing
    cache.fac_k = nothing
    return nothing
end

@inline _time_capacity(gf::GreenFunction) = size(gf, ndims(gf))

@inline function _ensure_gf_capacity!(gf::GreenFunction, tlen::Int)
    if _time_capacity(gf) < tlen
        resize!(gf, tlen)
    end
    return gf
end

function _set_identity_minus_scaled!(lhs::Matrix{ComplexF64}, op::Matrix{ComplexF64}, scale)
    @inbounds for j in axes(lhs, 2), i in axes(lhs, 1)
        lhs[i, j] = -scale * op[i, j]
    end
    @inbounds for i in axes(lhs, 1)
        lhs[i, i] += 1.0 + 0.0im
    end
    return lhs
end

@inline function _spectral_positive(omega::Real, bath::BathSpectrum)
    x = abs(omega)
    return bath.gamma * bath.omega_c^(1 - bath.s) * x^bath.s * exp(-x / bath.omega_c)
end

@inline function _spectral_antisym(omega::Real, bath::BathSpectrum)
    if omega > 0
        return _spectral_positive(omega, bath)
    elseif omega < 0
        return -_spectral_positive(-omega, bath)
    else
        return 0.0
    end
end

@inline function _coth_stable(x::Real)
    ax = abs(x)
    if ax < 1e-6
        # odd Laurent/Taylor continuation around x=0
        return inv(x) + x / 3 - x^3 / 45
    end
    return coth(x)
end

@inline function _bose_factor_coth(omega::Real, temperature::Real)
    abs(omega) < 1e-12 && return 0.0
    if temperature <= 0
        return sign(omega)
    end
    x = omega / (2 * temperature)
    return _coth_stable(x)
end

function _make_integral_problem(f::Function, a::Real, b::Real)
    # Handle API differences across Integrals.jl versions.
    try
        return Integrals.IntegralProblem((x, p) -> f(x), (Float64(a), Float64(b)), nothing)
    catch
        try
            return Integrals.IntegralProblem((x, p) -> f(x), (Float64(a), Float64(b)))
        catch
            return Integrals.IntegralProblem((x, p) -> f(x), Float64(a), Float64(b))
        end
    end
end

function _solve_integral(prob; abstol::Real, reltol::Real)
    alg_syms = (:QuadGKJL, :HCubatureJL, :CubaCuhre)
    last_err = nothing
    for sym in alg_syms
        if isdefined(Integrals, sym)
            alg_ctor = getfield(Integrals, sym)
            try
                sol = Integrals.solve(prob, alg_ctor(); abstol = abstol, reltol = reltol)
                return sol.u
            catch err
                last_err = err
            end
        end
    end
    if last_err === nothing
        error("No compatible Integrals.jl algorithm found (expected one of QuadGKJL/HCubatureJL/CubaCuhre).")
    end
    rethrow(last_err)
end

function _integrate_signed_frequency(integrand::Function, omega_max::Real; abstol::Real, reltol::Real)
    prob = _make_integral_problem(integrand, -omega_max, omega_max)
    return ComplexF64(_solve_integral(prob; abstol = abstol, reltol = reltol))
end

function _xi_component_tau(
    tau::Real,
    baths::Vector{BathSpectrum},
    is_keldysh::Bool,
    kernel_abstol::Real,
    kernel_reltol::Real,
)
    value = 0.0 + 0.0im
    for bath in baths
        integrand = if is_keldysh
            omega -> begin
                j = _spectral_antisym(omega, bath)
                c = _bose_factor_coth(omega, bath.temperature)
                return -0.5im / pi * j * c * exp(-im * omega * tau)
            end
        else
            omega -> begin
                j = _spectral_antisym(omega, bath)
                return -0.5im / pi * j * exp(-im * omega * tau)
            end
        end
        value += _integrate_signed_frequency(integrand, bath.omega_max; abstol = kernel_abstol, reltol = kernel_reltol)
    end
    return value
end

function _build_complex_interpolation(tau_grid::Vector{Float64}, values::Vector{ComplexF64})
    # Required by the request: use Interpolations.jl, no custom interpolator.
    return Interpolations.linear_interpolation(tau_grid, values; extrapolation_bc = Interpolations.Flat())
end

function precompute_bath_kernels(params::OpenSpinsParameters)
    ndiss = length(params.dissipative_spins)
    spin_to_diss = zeros(Int, params.n_spins)
    for (d, n) in enumerate(params.dissipative_spins)
        spin_to_diss[n] = d
    end

    grouped = [BathSpectrum[] for _ in 1:ndiss, _ in 1:N_COMP]
    has_channel = falses(ndiss, N_COMP)
    for bath in params.baths
        d = spin_to_diss[bath.spin]
        d == 0 && error("Bath attached to non-dissipative spin $(bath.spin).")
        push!(grouped[d, bath.component], bath)
        has_channel[d, bath.component] = true
    end

    tau_grid = collect(range(-params.tmax, params.tmax; length = params.kernel_ntau))
    i0 = findfirst(t -> t >= 0.0, tau_grid)
    i0 === nothing && error("Failed to build non-negative tau grid.")
    tau_nonneg = tau_grid[i0:end]
    xiK_table = zeros(ComplexF64, ndiss, N_COMP, length(tau_grid))
    xis_table = zeros(ComplexF64, ndiss, N_COMP, length(tau_grid))
    xiK_nonneg = zeros(ComplexF64, ndiss, N_COMP, length(tau_nonneg))
    xis_nonneg = zeros(ComplexF64, ndiss, N_COMP, length(tau_nonneg))

    for d in 1:ndiss, alpha in 1:N_COMP
        baths = grouped[d, alpha]
        isempty(baths) && continue
        for (j, tau) in pairs(tau_nonneg)
            valK = _xi_component_tau(
                tau,
                baths,
                true,
                params.kernel_abstol,
                params.kernel_reltol,
            )
            vals = _xi_component_tau(
                tau,
                baths,
                false,
                params.kernel_abstol,
                params.kernel_reltol,
            )

            if iszero(tau)
                # Xi^s(0) must vanish for antisymmetric spectral density.
                valK = 0.5 * (valK - conj(valK))
                vals = 0.0 + 0.0im
            end

            xiK_nonneg[d, alpha, j] = valK
            xis_nonneg[d, alpha, j] = vals

            ip = i0 + j - 1
            ineg = length(tau_grid) - ip + 1

            xiK_table[d, alpha, ip] = valK
            xis_table[d, alpha, ip] = vals

            if ineg != ip
                xiK_table[d, alpha, ineg] = -conj(valK)
                xis_table[d, alpha, ineg] = -conj(vals)
            end
        end
    end

    xiK_interp = [_build_complex_interpolation(tau_nonneg, vec(xiK_nonneg[d, alpha, :])) for d in 1:ndiss, alpha in 1:N_COMP]
    xis_interp = [_build_complex_interpolation(tau_nonneg, vec(xis_nonneg[d, alpha, :])) for d in 1:ndiss, alpha in 1:N_COMP]

    return KernelBundle(
        tau_grid,
        tau_nonneg,
        xiK_table,
        xis_table,
        xiK_interp,
        xis_interp,
        spin_to_diss,
        copy(params.dissipative_spins),
        has_channel,
    )
end

# -----------------------------------------------------------------------------
# Internal state
# -----------------------------------------------------------------------------

mutable struct SimulationState{I, G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12}
    params::OpenSpinsParameters
    kernels::KernelBundle{I}

    gK::G1
    gs::G2
    sigmaK::G3
    sigmas::G4
    piK::G5
    pis::G6
    omegaK::G7
    omegas::G8

    mcheckK::G9
    mchecks::G10
    dK::G11
    ds::G12

    trace_history::Array{ComplexF64, 3}   # (n, alpha, t)
    lambda_bar::Array{ComplexF64, 3}      # (n, alpha, t)
    Lambda_bar::Array{ComplexF64, 3}      # (n, alpha, t)
    heff::Array{ComplexF64, 4}            # (n, a, b, t)
    spin_spin_scale::Vector{Float64}
    spin_bath_scale::Vector{Float64}
    scratch::Vector{OpenSpinsScratch}
    mcheck_endpoint_cache::McheckEndpointCache
    weight_history::Vector{Vector{Float64}}  # adaptive quadrature weights per upper index
end

function _initial_gf_slices(params::OpenSpinsParameters, init::OpenSpinsInitialState)
    size(init.spin_expectation) == (params.n_spins, N_COMP) ||
        error("Initial spin_expectation must have shape (n_spins, 3).")

    gK00 = zeros(ComplexF64, params.n_spins, N_FIELD, N_FIELD)
    gs00 = zeros(ComplexF64, params.n_spins, N_FIELD, N_FIELD)

    id4 = Matrix{ComplexF64}(I, N_FIELD, N_FIELD)
    for n in 1:params.n_spins
        rhs = 2 * (init.spin_length + 0.5) * id4
        for alpha in 1:N_COMP
            rhs .+= 2 * init.spin_expectation[n, alpha] * K_MATRICES[alpha]
        end
        gK00[n, :, :] .= -im .* rhs
        gs00[n, :, :] .= im .* K0_MATRIX
    end

    return gK00, gs00
end

function initialize_state(params::OpenSpinsParameters, kernels::KernelBundle, init::OpenSpinsInitialState)
    ndiss = length(params.dissipative_spins)
    spin_spin_scale = [params.spin_spin_profile(0.0)]
    spin_bath_scale = [params.spin_bath_profile(0.0)]

    gK = GreenFunction(zeros(ComplexF64, params.n_spins, N_FIELD, N_FIELD, 1, 1), SymmetricSiteField)
    gs = GreenFunction(zeros(ComplexF64, params.n_spins, N_FIELD, N_FIELD, 1, 1), AntiSymmetricSiteField)
    sigmaK = GreenFunction(zeros(ComplexF64, params.n_spins, N_FIELD, N_FIELD, 1, 1), SymmetricSiteField)
    sigmas = GreenFunction(zeros(ComplexF64, params.n_spins, N_FIELD, N_FIELD, 1, 1), AntiSymmetricSiteField)

    piK = GreenFunction(zeros(ComplexF64, ndiss, N_COMP, 1, 1), SymmetricScalarChannel)
    pis = GreenFunction(zeros(ComplexF64, ndiss, N_COMP, 1, 1), AntiSymmetricScalarChannel)

    omegaK = GreenFunction(zeros(ComplexF64, params.n_spins, N_COMP, N_COMP, 1, 1), SymmetricSiteField)
    omegas = GreenFunction(zeros(ComplexF64, params.n_spins, N_COMP, N_COMP, 1, 1), AntiSymmetricSiteField)

    mcheckK = GreenFunction(zeros(ComplexF64, params.n_spins, params.n_spins, N_COMP, N_COMP, 1, 1), SymmetricMeanFieldTensor)
    mchecks = GreenFunction(zeros(ComplexF64, params.n_spins, params.n_spins, N_COMP, N_COMP, 1, 1), AntiSymmetricMeanFieldTensor)

    dK = GreenFunction(zeros(ComplexF64, ndiss, N_COMP, 1, 1), SymmetricScalarChannel)
    ds = GreenFunction(zeros(ComplexF64, ndiss, N_COMP, 1, 1), AntiSymmetricScalarChannel)

    gK00, gs00 = _initial_gf_slices(params, init)
    gK[1, 1] = gK00
    gs[1, 1] = gs00

    dK00 = zeros(ComplexF64, ndiss, N_COMP)
    ds00 = zeros(ComplexF64, ndiss, N_COMP)
    for d in 1:ndiss, alpha in 1:N_COMP
        if kernels.has_channel[d, alpha]
            bath_scale_sq = spin_bath_scale[1]^2
            dK00[d, alpha] = 2 * bath_scale_sq * kernels.xiK_interp[d, alpha](0.0)
            ds00[d, alpha] = 2 * bath_scale_sq * kernels.xis_interp[d, alpha](0.0)
        end
    end
    dK[1, 1] = dK00
    ds[1, 1] = ds00

    trace_history = zeros(ComplexF64, params.n_spins, N_COMP, 1)
    lambda_bar = zeros(ComplexF64, params.n_spins, N_COMP, 1)
    Lambda_bar = zeros(ComplexF64, params.n_spins, N_COMP, 1)
    heff = zeros(ComplexF64, params.n_spins, N_FIELD, N_FIELD, 1)

    return SimulationState(
        params,
        kernels,
        gK,
        gs,
        sigmaK,
        sigmas,
        piK,
        pis,
        omegaK,
        omegas,
        mcheckK,
        mchecks,
        dK,
        ds,
        trace_history,
        lambda_bar,
        Lambda_bar,
        heff,
        spin_spin_scale,
        spin_bath_scale,
        _make_thread_scratch(params.n_spins),
        McheckEndpointCache(),
        [Float64[]],
    )
end

function _ensure_point_capacity!(state::SimulationState, tlen::Int)
    target = max(tlen, _time_capacity(state.gK), _time_capacity(state.gs))
    _ensure_gf_capacity!(state.sigmaK, target)
    _ensure_gf_capacity!(state.sigmas, target)
    _ensure_gf_capacity!(state.piK, target)
    _ensure_gf_capacity!(state.pis, target)
    _ensure_gf_capacity!(state.omegaK, target)
    _ensure_gf_capacity!(state.omegas, target)
    _ensure_gf_capacity!(state.mcheckK, target)
    _ensure_gf_capacity!(state.mchecks, target)
    _ensure_gf_capacity!(state.dK, target)
    _ensure_gf_capacity!(state.ds, target)

    if size(state.trace_history, 3) < target
        old = size(state.trace_history, 3)
        n = state.params.n_spins
        tr_new = zeros(ComplexF64, n, N_COMP, target)
        lb_new = zeros(ComplexF64, n, N_COMP, target)
        Lb_new = zeros(ComplexF64, n, N_COMP, target)
        hf_new = zeros(ComplexF64, n, N_FIELD, N_FIELD, target)

        tr_new[:, :, 1:old] .= state.trace_history
        lb_new[:, :, 1:old] .= state.lambda_bar
        Lb_new[:, :, 1:old] .= state.Lambda_bar
        hf_new[:, :, :, 1:old] .= state.heff

        state.trace_history = tr_new
        state.lambda_bar = lb_new
        state.Lambda_bar = Lb_new
        state.heff = hf_new
    end
    if length(state.spin_spin_scale) < target
        resize!(state.spin_spin_scale, target)
        resize!(state.spin_bath_scale, target)
    end
    return nothing
end

@inline _bath_data(bath::BathSpectrum) = (
    spin = bath.spin,
    component = bath.component,
    gamma = bath.gamma,
    s = bath.s,
    omega_c = bath.omega_c,
    temperature = bath.temperature,
    omega_max = bath.omega_max,
    n_omega = bath.n_omega,
)

function _parameters_data(params::OpenSpinsParameters)
    return (
        n_spins = params.n_spins,
        h = copy(params.h),
        J = copy(params.J),
        dissipative_spins = copy(params.dissipative_spins),
        baths = [_bath_data(bath) for bath in params.baths],
        spin_spin_profile = interaction_profile_data(params.spin_spin_profile),
        spin_bath_profile = interaction_profile_data(params.spin_bath_profile),
        tmax = params.tmax,
        dtini = params.dtini,
        dtmax = params.dtmax,
        atol = params.atol,
        rtol = params.rtol,
        kernel_ntau = params.kernel_ntau,
        kernel_abstol = params.kernel_abstol,
        kernel_reltol = params.kernel_reltol,
        symmetry_tol = params.symmetry_tol,
        output_basename = params.output_basename,
    )
end

function OpenSpinsParameters(data::NamedTuple)
    baths = [
        BathSpectrum(
            bath.spin,
            bath.component;
            gamma = bath.gamma,
            s = bath.s,
            omega_c = bath.omega_c,
            temperature = bath.temperature,
            omega_max = bath.omega_max,
            n_omega = bath.n_omega,
        ) for bath in data.baths
    ]
    return OpenSpinsParameters(
        n_spins = data.n_spins,
        h = data.h,
        J = data.J,
        dissipative_spins = data.dissipative_spins,
        baths = baths,
        spin_spin_profile = InteractionProfile(data.spin_spin_profile),
        spin_bath_profile = InteractionProfile(data.spin_bath_profile),
        tmax = data.tmax,
        dtini = data.dtini,
        dtmax = data.dtmax,
        atol = data.atol,
        rtol = data.rtol,
        kernel_ntau = data.kernel_ntau,
        kernel_abstol = data.kernel_abstol,
        kernel_reltol = data.kernel_reltol,
        symmetry_tol = data.symmetry_tol,
        output_basename = data.output_basename,
    )
end

function _trim_point_capacity!(state::SimulationState, tlen::Int)
    _time_capacity(state.sigmaK) == tlen || resize!(state.sigmaK, tlen)
    _time_capacity(state.sigmas) == tlen || resize!(state.sigmas, tlen)
    _time_capacity(state.piK) == tlen || resize!(state.piK, tlen)
    _time_capacity(state.pis) == tlen || resize!(state.pis, tlen)
    _time_capacity(state.omegaK) == tlen || resize!(state.omegaK, tlen)
    _time_capacity(state.omegas) == tlen || resize!(state.omegas, tlen)
    _time_capacity(state.mcheckK) == tlen || resize!(state.mcheckK, tlen)
    _time_capacity(state.mchecks) == tlen || resize!(state.mchecks, tlen)
    _time_capacity(state.dK) == tlen || resize!(state.dK, tlen)
    _time_capacity(state.ds) == tlen || resize!(state.ds, tlen)

    if size(state.trace_history, 3) != tlen
        state.trace_history = state.trace_history[:, :, 1:tlen]
        state.lambda_bar = state.lambda_bar[:, :, 1:tlen]
        state.Lambda_bar = state.Lambda_bar[:, :, 1:tlen]
        state.heff = state.heff[:, :, :, 1:tlen]
    end
    resize!(state.spin_spin_scale, tlen)
    resize!(state.spin_bath_scale, tlen)
    return nothing
end

@inline function _sync_profile_history!(state::SimulationState, ts, t1::Int, t2::Int)
    state.spin_spin_scale[t1] = state.params.spin_spin_profile(ts[t1])
    state.spin_bath_scale[t1] = state.params.spin_bath_profile(ts[t1])
    state.spin_spin_scale[t2] = state.params.spin_spin_profile(ts[t2])
    state.spin_bath_scale[t2] = state.params.spin_bath_profile(ts[t2])
    return nothing
end

@inline function _ensure_weight_slot!(state::SimulationState, idx::Int)
    while length(state.weight_history) < idx
        push!(state.weight_history, Float64[])
    end
    return nothing
end

@inline function _sync_weight_history!(state::SimulationState, w1, w2, t1::Int, t2::Int)
    _ensure_weight_slot!(state, t1)
    _ensure_weight_slot!(state, t2)
    state.weight_history[t1] = w1
    state.weight_history[t2] = w2
    return nothing
end

@inline function _weights_at(state::SimulationState, tidx::Int)
    tidx <= length(state.weight_history) || error("Missing quadrature weights for index $tidx.")
    ws = state.weight_history[tidx]
    isempty(ws) && error("Quadrature weights at index $tidx are not initialized.")
    return ws
end

@inline function _enforce_gs_equal_time_commutator!(state::SimulationState, t::Int)
    gs_data = state.gs.data
    @inbounds for n in 1:state.params.n_spins, i in 1:N_FIELD, j in 1:N_FIELD
        gs_data[n, i, j, t, t] = gs_data[n, i, j, 1, 1]
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Tensor indexing helpers
# -----------------------------------------------------------------------------

@inline function _trace_ka_x_kb_yt(xdata, ydata, n::Int, t1::Int, t2::Int, alpha::Int, beta::Int)
    stencil = TRACE_STENCILS[alpha, beta]
    acc = 0.0 + 0.0im
    @inbounds for term in stencil
        acc += term.coeff * xdata[n, term.j, term.k, t1, t2] * ydata[n, term.i, term.l, t1, t2]
    end
    return acc
end

@inline function _jmj_diag_entry(
    m_data,
    J::Array{Float64, 4},
    n::Int,
    alpha::Int,
    beta::Int,
    t1::Int,
    t2::Int,
)
    n_spins = size(J, 1)
    acc = 0.0 + 0.0im
    @inbounds for m in 1:n_spins, delta in 1:N_COMP, p in 1:n_spins, eta in 1:N_COMP
        j1 = J[n, m, alpha, delta]
        iszero(j1) && continue
        j2 = J[p, n, eta, beta]
        iszero(j2) && continue
        acc += j1 * m_data[m, p, delta, eta, t1, t2] * j2
    end
    return acc
end

@inline function _dressed_jmj_diag_entry(
    m_data,
    J::Array{Float64, 4},
    n::Int,
    alpha::Int,
    beta::Int,
    t1::Int,
    t2::Int,
    left_scale::Real,
    right_scale::Real,
)
    return left_scale * right_scale * _jmj_diag_entry(m_data, J, n, alpha, beta, t1, t2)
end

# -----------------------------------------------------------------------------
# Equation blocks
# -----------------------------------------------------------------------------

function _update_polarizations!(state::SimulationState, t1::Int, t2::Int)
    n_spins = state.params.n_spins
    ndiss = length(state.params.dissipative_spins)
    gK_data = state.gK.data
    gs_data = state.gs.data
    omegaK_data = state.omegaK.data
    omegas_data = state.omegas.data
    piK_data = state.piK.data
    pis_data = state.pis.data

    @inbounds for d in 1:ndiss, alpha in 1:N_COMP
        piK_data[d, alpha, t1, t2] = 0.0 + 0.0im
        pis_data[d, alpha, t1, t2] = 0.0 + 0.0im
        if t1 != t2
            piK_data[d, alpha, t2, t1] = 0.0 + 0.0im
            pis_data[d, alpha, t2, t1] = 0.0 + 0.0im
        end
    end

    @inbounds for n in 1:n_spins
        d = state.kernels.spin_to_diss[n]
        for alpha in 1:N_COMP
            for beta in 1:N_COMP
                tK = _trace_ka_x_kb_yt(gK_data, gK_data, n, t1, t2, alpha, beta) +
                     _trace_ka_x_kb_yt(gs_data, gs_data, n, t1, t2, alpha, beta)
                ts = _trace_ka_x_kb_yt(gK_data, gs_data, n, t1, t2, alpha, beta)
                valK = 0.0625im * tK
                vals = 0.125im * ts

                omegaK_data[n, alpha, beta, t1, t2] = valK
                omegas_data[n, alpha, beta, t1, t2] = vals

                if t1 != t2
                    omegaK_data[n, beta, alpha, t2, t1] = valK
                    omegas_data[n, beta, alpha, t2, t1] = -vals
                end
            end

            if d != 0
                tK = _trace_ka_x_kb_yt(gK_data, gK_data, n, t1, t2, alpha, alpha) +
                     _trace_ka_x_kb_yt(gs_data, gs_data, n, t1, t2, alpha, alpha)
                ts = _trace_ka_x_kb_yt(gK_data, gs_data, n, t1, t2, alpha, alpha)
                pik = 0.0625im * tK
                pis = 0.125im * ts
                piK_data[d, alpha, t1, t2] = pik
                pis_data[d, alpha, t1, t2] = pis
                if t1 != t2
                    piK_data[d, alpha, t2, t1] = pik
                    pis_data[d, alpha, t2, t1] = -pis
                end
            end
        end
    end

    return nothing
end

function _update_meanfield_propagator!(state::SimulationState, _ts::Vector{<:Real}, w1, w2, t1::Int, t2::Int)
    n_spins = state.params.n_spins
    J = state.params.J
    mchecks_data = state.mchecks.data
    mcheckK_data = state.mcheckK.data
    omegas_data = state.omegas.data
    omegaK_data = state.omegaK.data
    scratch = _scratch(state)

    equal_time = (t1 == t2)
    # For t=t', the spectral Volterra interval is empty and its endpoint term must vanish exactly.
    cached_fac_s, fac_k = _mcheck_endpoint_factors!(state, w1, t1)
    fac_s = equal_time ? nothing : cached_fac_s

    rhs = scratch.rhs
    sol = scratch.sol

    @inbounds for np in 1:n_spins, beta in 1:N_COMP
        fill!(rhs, 0.0 + 0.0im)
        for n in 1:n_spins, alpha in 1:N_COMP
            row = _na_index(n, alpha)
            accs = 0.0 + 0.0im
            if !equal_time
                accs_hi = 0.0 + 0.0im
                for s in 1:(t1 - 1)
                    exchange_scale = state.spin_spin_scale[s]
                    inner = 0.0 + 0.0im
                    for gamma in 1:N_COMP, m in 1:n_spins, delta in 1:N_COMP
                        jval = exchange_scale * J[n, m, gamma, delta]
                        iszero(jval) && continue
                        inner += omegas_data[n, alpha, gamma, t1, s] * jval * mchecks_data[m, np, delta, beta, s, t2]
                    end
                    accs_hi += w1[s] * inner
                end

                accs_lo = 0.0 + 0.0im
                for s in 1:t2
                    exchange_scale = state.spin_spin_scale[s]
                    inner = 0.0 + 0.0im
                    for gamma in 1:N_COMP, m in 1:n_spins, delta in 1:N_COMP
                        jval = exchange_scale * J[n, m, gamma, delta]
                        iszero(jval) && continue
                        inner += omegas_data[n, alpha, gamma, t1, s] * jval * mchecks_data[m, np, delta, beta, s, t2]
                    end
                    accs_lo += w2[s] * inner
                end
                accs = accs_hi - accs_lo
            end

            rhs[row] = ((n == np) ? (4 * omegas_data[n, alpha, beta, t1, t2]) : (0.0 + 0.0im)) + 2 * accs
        end

        if fac_s === nothing
            sol .= rhs
        else
            ldiv!(sol, fac_s, rhs)
        end

        for n in 1:n_spins, alpha in 1:N_COMP
            row = _na_index(n, alpha)
            vals = sol[row]
            mchecks_data[n, np, alpha, beta, t1, t2] = vals
            if t1 != t2
                mchecks_data[np, n, beta, alpha, t2, t1] = -vals
            end
        end
    end

    @inbounds for np in 1:n_spins, beta in 1:N_COMP
        fill!(rhs, 0.0 + 0.0im)
        for n in 1:n_spins, alpha in 1:N_COMP
            row = _na_index(n, alpha)
            accK1 = 0.0 + 0.0im
            for s in 1:(t1 - 1)
                exchange_scale = state.spin_spin_scale[s]
                inner = 0.0 + 0.0im
                for gamma in 1:N_COMP, m in 1:n_spins, delta in 1:N_COMP
                    jval = exchange_scale * J[n, m, gamma, delta]
                    iszero(jval) && continue
                    inner += omegas_data[n, alpha, gamma, t1, s] * jval * mcheckK_data[m, np, delta, beta, s, t2]
                end
                accK1 += w1[s] * inner
            end

            accK2 = 0.0 + 0.0im
            for s in 1:t2
                exchange_scale = state.spin_spin_scale[s]
                inner = 0.0 + 0.0im
                for gamma in 1:N_COMP, m in 1:n_spins, delta in 1:N_COMP
                    jval = exchange_scale * J[n, m, gamma, delta]
                    iszero(jval) && continue
                    inner += omegaK_data[n, alpha, gamma, t1, s] * jval * mchecks_data[m, np, delta, beta, s, t2]
                end
                accK2 += w2[s] * inner
            end

            rhs[row] = ((n == np) ? (4 * omegaK_data[n, alpha, beta, t1, t2]) : (0.0 + 0.0im)) + 2 * accK1 - 2 * accK2
        end

        if fac_k === nothing
            sol .= rhs
        else
            ldiv!(sol, fac_k, rhs)
        end

        for n in 1:n_spins, alpha in 1:N_COMP
            row = _na_index(n, alpha)
            valK = sol[row]
            mcheckK_data[n, np, alpha, beta, t1, t2] = valK
            if t1 != t2
                mcheckK_data[np, n, beta, alpha, t2, t1] = valK
            end
        end
    end

    return nothing
end

@inline function _xiK_base(state::SimulationState, d::Int, alpha::Int, tau::Real)
    t = Float64(tau)
    if t >= 0
        return state.kernels.xiK_interp[d, alpha](t)
    end
    return -conj(state.kernels.xiK_interp[d, alpha](-t))
end

@inline function _xis_base(state::SimulationState, d::Int, alpha::Int, tau::Real)
    t = Float64(tau)
    if t >= 0
        return state.kernels.xis_interp[d, alpha](t)
    end
    return -conj(state.kernels.xis_interp[d, alpha](-t))
end

@inline function _xiK(state::SimulationState, d::Int, alpha::Int, ts, t1::Int, t2::Int)
    scale = state.spin_bath_scale[t1] * state.spin_bath_scale[t2]
    return scale * _xiK_base(state, d, alpha, ts[t1] - ts[t2])
end

@inline function _xis(state::SimulationState, d::Int, alpha::Int, ts, t1::Int, t2::Int)
    scale = state.spin_bath_scale[t1] * state.spin_bath_scale[t2]
    return scale * _xis_base(state, d, alpha, ts[t1] - ts[t2])
end

@inline function _solve_implicit_endpoint(rhs::ComplexF64, coeff::ComplexF64; eps::Float64 = 1e-12)
    denom = 1.0 + 0.0im - coeff
    abs(denom) > eps || error("Implicit bath propagator endpoint solve is near-singular (|1-coeff|=$(abs(denom))).")
    return rhs / denom
end

@inline _na_index(n::Int, alpha::Int) = (n - 1) * N_COMP + alpha

function _build_mcheck_endpoint_operator!(op::Matrix{ComplexF64}, state::SimulationState, t1::Int)
    n_spins = state.params.n_spins
    J = state.params.J
    omegas_data = state.omegas.data
    fill!(op, 0.0 + 0.0im)

    @inbounds for n in 1:n_spins, alpha in 1:N_COMP
        row = _na_index(n, alpha)
        for gamma in 1:N_COMP
            omega = omegas_data[n, alpha, gamma, t1, t1]
            iszero(omega) && continue
            for m in 1:n_spins, delta in 1:N_COMP
                jval = J[n, m, gamma, delta]
                iszero(jval) && continue
                col = _na_index(m, delta)
                op[row, col] += omega * jval
            end
        end
    end

    return nothing
end

function _mcheck_endpoint_factors!(state::SimulationState, w1, t1::Int)
    cache = state.mcheck_endpoint_cache
    scale_s = Float64(2 * w1[t1] * state.spin_spin_scale[t1])
    scale_k = scale_s

    if cache.row == t1 && cache.scale_s == scale_s && cache.scale_k == scale_k
        return cache.fac_s, cache.fac_k
    end

    scratch = _scratch(state)
    endpoint_op = scratch.endpoint_op
    _build_mcheck_endpoint_operator!(endpoint_op, state, t1)

    fac_s = nothing
    if abs(scale_s) > 1e-12
        lhs_s = _set_identity_minus_scaled!(scratch.lhs_s, endpoint_op, scale_s)
        fac_s = lu!(lhs_s)
    end

    fac_k = nothing
    if abs(scale_k) > 1e-12
        lhs_k = _set_identity_minus_scaled!(scratch.lhs_k, endpoint_op, scale_k)
        fac_k = lu!(lhs_k)
    end

    cache.row = t1
    cache.scale_s = scale_s
    cache.scale_k = scale_k
    cache.fac_s = fac_s
    cache.fac_k = fac_k
    return fac_s, fac_k
end

function _update_bath_propagator!(state::SimulationState, ts::Vector{<:Real}, w1, w2, t1::Int, t2::Int)
    ndiss = length(state.params.dissipative_spins)
    dK_data = state.dK.data
    ds_data = state.ds.data
    piK_data = state.piK.data
    pis_data = state.pis.data

    for d in 1:ndiss, alpha in 1:N_COMP
        if !state.kernels.has_channel[d, alpha]
            dK_data[d, alpha, t1, t2] = 0.0 + 0.0im
            ds_data[d, alpha, t1, t2] = 0.0 + 0.0im
            if t1 != t2
                dK_data[d, alpha, t2, t1] = 0.0 + 0.0im
                ds_data[d, alpha, t2, t1] = 0.0 + 0.0im
            end
            continue
        end

        xiK_t = _xiK(state, d, alpha, ts, t1, t2)
        xis_t = _xis(state, d, alpha, ts, t1, t2)
        xis_0 = _xis(state, d, alpha, ts, t1, t1)
        w_t1 = _weights_at(state, t1)
        w_t2 = _weights_at(state, t2)
        has_endpoint_xis = !iszero(xis_0)
        endpoint_coeff = has_endpoint_xis ? 2 * w1[t1] * w_t1[t1] * xis_0 * pis_data[d, alpha, t1, t1] : 0.0 + 0.0im

        # D^s is explicit in the current antisymmetric-kernel setup (`Xi^s(0)=0`),
        # so no local implicit endpoint solve is needed for this block.
        # For t=t', the double spectral integral is empty and must be zero exactly.
        accs = 0.0 + 0.0im
        if t1 != t2
            # Integral partition for shifted lower bounds:
            # ∫_{t'}^{t}∫_{t'}^{t1} = ∫_{0}^{t}∫_{0}^{t1} - ∫_{0}^{t}∫_{0}^{t'} - ∫_{0}^{t'}∫_{0}^{t1} + ∫_{0}^{t'}∫_{0}^{t'}.
            i00 = 0.0 + 0.0im
            for s1 in 1:(t1 - 1)
                ws1 = _weights_at(state, s1)
                xis_t1s1 = _xis(state, d, alpha, ts, t1, s1)
                inner = 0.0 + 0.0im
                for s2 in 1:s1
                    inner += ws1[s2] * pis_data[d, alpha, s1, s2] * ds_data[d, alpha, s2, t2]
                end
                i00 += w1[s1] * xis_t1s1 * inner
            end
            if has_endpoint_xis
                inner_last_s = 0.0 + 0.0im
                for s2 in 1:(t1 - 1)
                    inner_last_s += w_t1[s2] * pis_data[d, alpha, t1, s2] * ds_data[d, alpha, s2, t2]
                end
                i00 += w1[t1] * xis_0 * inner_last_s
            end

            i01 = 0.0 + 0.0im
            i01_tmax = has_endpoint_xis ? t1 : (t1 - 1)
            for s1 in 1:i01_tmax
                xis_t1s1 = _xis(state, d, alpha, ts, t1, s1)
                inner = 0.0 + 0.0im
                for s2 in 1:t2
                    inner += w_t2[s2] * pis_data[d, alpha, s1, s2] * ds_data[d, alpha, s2, t2]
                end
                i01 += w1[s1] * xis_t1s1 * inner
            end

            i10 = 0.0 + 0.0im
            for s1 in 1:t2
                ws1 = _weights_at(state, s1)
                xis_t1s1 = _xis(state, d, alpha, ts, t1, s1)
                inner = 0.0 + 0.0im
                for s2 in 1:s1
                    inner += ws1[s2] * pis_data[d, alpha, s1, s2] * ds_data[d, alpha, s2, t2]
                end
                i10 += w2[s1] * xis_t1s1 * inner
            end

            i11 = 0.0 + 0.0im
            for s1 in 1:t2
                xis_t1s1 = _xis(state, d, alpha, ts, t1, s1)
                inner = 0.0 + 0.0im
                for s2 in 1:t2
                    inner += w_t2[s2] * pis_data[d, alpha, s1, s2] * ds_data[d, alpha, s2, t2]
                end
                i11 += w2[s1] * xis_t1s1 * inner
            end

            accs = i00 - i01 - i10 + i11
        end

        ds_rhs = 2 * xis_t + 2 * accs
        ds_val = ds_rhs
        ds_data[d, alpha, t1, t2] = ds_val
        if t1 != t2
            ds_data[d, alpha, t2, t1] = -ds_val
        end

        # D^K implicit endpoint solve with the same endpoint coefficient.
        acc1 = 0.0 + 0.0im
        for s1 in 1:(t1 - 1)
            ws1 = _weights_at(state, s1)
            xis_t1s1 = _xis(state, d, alpha, ts, t1, s1)
            inner = 0.0 + 0.0im
            for s2 in 1:s1
                inner += ws1[s2] * pis_data[d, alpha, s1, s2] * dK_data[d, alpha, s2, t2]
            end
            acc1 += w1[s1] * xis_t1s1 * inner
        end
        if has_endpoint_xis
            inner_last_k = 0.0 + 0.0im
            for s2 in 1:(t1 - 1)
                inner_last_k += w_t1[s2] * pis_data[d, alpha, t1, s2] * dK_data[d, alpha, s2, t2]
            end
            acc1 += w1[t1] * xis_0 * inner_last_k
        end

        acc2 = 0.0 + 0.0im
        for s1 in 1:t2
            ws1 = _weights_at(state, s1)
            inner = 0.0 + 0.0im
            for s2 in 1:s1
                inner += ws1[s2] * _xiK(state, d, alpha, ts, t1, s2) * pis_data[d, alpha, s2, s1]
            end
            acc2 += w2[s1] * inner * ds_data[d, alpha, s1, t2]
        end

        acc3 = 0.0 + 0.0im
        acc3_tmax = has_endpoint_xis ? t1 : (t1 - 1)
        for s1 in 1:acc3_tmax
            xis_t1s1 = _xis(state, d, alpha, ts, t1, s1)
            inner = 0.0 + 0.0im
            for s2 in 1:t2
                inner += w_t2[s2] * piK_data[d, alpha, s1, s2] * ds_data[d, alpha, s2, t2]
            end
            acc3 += w1[s1] * xis_t1s1 * inner
        end

        dK_rhs = 2 * xiK_t + 2 * acc1 + 2 * acc2 - 2 * acc3
        dK_val = _solve_implicit_endpoint(dK_rhs, endpoint_coeff)

        dK_data[d, alpha, t1, t2] = dK_val
        if t1 != t2
            dK_data[d, alpha, t2, t1] = dK_val
        end
    end

    return nothing
end

function _update_self_energies!(state::SimulationState, t1::Int, t2::Int)
    n_spins = state.params.n_spins
    J = state.params.J
    gK_data = state.gK.data
    gs_data = state.gs.data
    mK_data = state.mcheckK.data
    ms_data = state.mchecks.data
    sigmaK_data = state.sigmaK.data
    sigmas_data = state.sigmas.data
    scratch = _scratch(state)
    jmjK = scratch.jmjK
    jmjs = scratch.jmjs
    left_exchange_scale = state.spin_spin_scale[t1]
    right_exchange_scale = state.spin_spin_scale[t2]

    for n in 1:n_spins
        d = state.kernels.spin_to_diss[n]
        @inbounds for alpha in 1:N_COMP, beta in 1:N_COMP
            jmjK[alpha, beta] = _dressed_jmj_diag_entry(
                mK_data, J, n, alpha, beta, t1, t2, left_exchange_scale, right_exchange_scale,
            )
            jmjs[alpha, beta] = _dressed_jmj_diag_entry(
                ms_data, J, n, alpha, beta, t1, t2, left_exchange_scale, right_exchange_scale,
            )
        end

        @inbounds for i in 1:N_FIELD, l in 1:N_FIELD
            sK = 0.0 + 0.0im
            ss = 0.0 + 0.0im

            for alpha in 1:N_COMP
                Ka = K_MATRICES[alpha]
                bathK = (d == 0) ? (0.0 + 0.0im) : state.dK.data[d, alpha, t1, t2]
                baths = (d == 0) ? (0.0 + 0.0im) : state.ds.data[d, alpha, t1, t2]

                gKg = 0.0 + 0.0im
                gsg = 0.0 + 0.0im
                for j in 1:N_FIELD, k in 1:N_FIELD
                    gKg += Ka[i, j] * gK_data[n, j, k, t1, t2] * Ka[k, l]
                    gsg += Ka[i, j] * gs_data[n, j, k, t1, t2] * Ka[k, l]
                end
                sK += 0.125im * (gKg * bathK + gsg * baths)
                ss += 0.125im * (gKg * baths + gsg * bathK)
            end

            for alpha in 1:N_COMP, beta in 1:N_COMP
                Ka = K_MATRICES[alpha]
                Kb = K_MATRICES[beta]
                mk = jmjK[alpha, beta]
                ms = jmjs[alpha, beta]

                gKg = 0.0 + 0.0im
                gsg = 0.0 + 0.0im
                for j in 1:N_FIELD, k in 1:N_FIELD
                    gKg += Ka[i, j] * gK_data[n, j, k, t1, t2] * Kb[k, l]
                    gsg += Ka[i, j] * gs_data[n, j, k, t1, t2] * Kb[k, l]
                end
                sK += 0.125im * (gKg * mk + gsg * ms)
                ss += 0.125im * (gKg * ms + gsg * mk)
            end

            sigmaK_data[n, i, l, t1, t2] = sK
            sigmas_data[n, i, l, t1, t2] = ss
            if t1 != t2
                sigmaK_data[n, l, i, t2, t1] = sK
                sigmas_data[n, l, i, t2, t1] = -ss
            end
        end
    end

    return nothing
end

function _update_fields_diagonal!(state::SimulationState, ts::Vector{<:Real}, wdiag, t::Int)
    n_spins = state.params.n_spins
    gK_data = state.gK.data
    tr_hist = state.trace_history
    lambda_bar = state.lambda_bar
    Lambda_bar = state.Lambda_bar
    heff_data = state.heff
    exchange_scale = state.spin_spin_scale[t]

    @inbounds for n in 1:n_spins, alpha in 1:N_COMP
        trval = 0.0 + 0.0im
        Ka = K_MATRICES[alpha]
        for i in 1:N_FIELD, j in 1:N_FIELD
            trval += Ka[i, j] * gK_data[n, j, i, t, t]
        end
        tr_hist[n, alpha, t] = trval
    end

    for n in 1:n_spins, alpha in 1:N_COMP
        x = 0.0 + 0.0im
        for beta in 1:N_COMP, np in 1:n_spins
            jval = exchange_scale * state.params.J[n, np, alpha, beta]
            iszero(jval) && continue
            x += jval * tr_hist[np, beta, t]
        end
        Lambda_bar[n, alpha, t] = 0.25im * x

        d = state.kernels.spin_to_diss[n]
        if d == 0 || !state.kernels.has_channel[d, alpha]
            lambda_bar[n, alpha, t] = 0.0 + 0.0im
        else
            y = 0.0 + 0.0im
            for s in 1:t
                y += wdiag[s] * _xis(state, d, alpha, ts, t, s) * tr_hist[n, alpha, s]
            end
            lambda_bar[n, alpha, t] = 0.25im * y
        end
    end

    @inbounds for n in 1:n_spins, i in 1:N_FIELD, j in 1:N_FIELD
        val = 0.0 + 0.0im
        for alpha in 1:N_COMP
            pref = (state.params.h[n, alpha] + lambda_bar[n, alpha, t] + Lambda_bar[n, alpha, t]) / 4
            val += pref * K_MATRICES[alpha][i, j]
        end
        heff_data[n, i, j, t] = val
    end

    return nothing
end

function _update_diagonal_endpoint_inputs!(state::SimulationState, t::Int)
    _enforce_gs_equal_time_commutator!(state, t)
    _update_polarizations!(state, t, t)
    _invalidate_mcheck_endpoint_cache!(state)
    return nothing
end

function _update_polarization_row!(state::SimulationState, t::Int)
    _enforce_gs_equal_time_commutator!(state, t)
    for s in 1:t
        _update_polarizations!(state, t, s)
    end
    _invalidate_mcheck_endpoint_cache!(state)
    return nothing
end

function _update_diagonal_auxiliaries!(state::SimulationState, ts::Vector{<:Real}, wdiag, t::Int)
    _update_diagonal_endpoint_inputs!(state, t)
    _update_meanfield_propagator!(state, ts, wdiag, wdiag, t, t)
    _update_bath_propagator!(state, ts, wdiag, wdiag, t, t)
    _update_self_energies!(state, t, t)
    _update_fields_diagonal!(state, ts, wdiag, t)
    return nothing
end

function _update_auxiliaries!(state::SimulationState, ts::Vector{<:Real}, w1, w2, t1::Int, t2::Int)
    _ensure_point_capacity!(state, length(ts))
    _sync_profile_history!(state, ts, t1, t2)
    _sync_weight_history!(state, w1, w2, t1, t2)

    if t2 == 1
        # M and D at any current-row point depend on Omega/Pi(t,s) across the
        # whole row.  KadanoffBaym invokes row callbacks in increasing t2.
        _update_polarization_row!(state, t1)
    end

    if t1 == t2
        # The Keldysh M and D diagonal equations depend on the current row's
        # off-diagonal values, so the full diagonal update must run last.
        _update_diagonal_auxiliaries!(state, ts, w1, t1)
    else
        _update_meanfield_propagator!(state, ts, w1, w2, t1, t2)
        _update_bath_propagator!(state, ts, w1, w2, t1, t2)
        _update_self_energies!(state, t1, t2)
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Kadanoff-Baym RHS
# -----------------------------------------------------------------------------

function _fv_arrays!(dgK::Array{ComplexF64, 3}, dgs::Array{ComplexF64, 3}, state::SimulationState, ts::Vector{<:Real}, w1, w2, t1::Int, t2::Int, scratch::OpenSpinsScratch; compute_spectral::Bool = true)
    n_spins = state.params.n_spins
    fill!(dgK, 0.0 + 0.0im)
    fill!(dgs, 0.0 + 0.0im)

    collK_1 = scratch.collK_1
    collK_2 = scratch.collK_2
    colls = scratch.colls

    @inbounds for n in 1:n_spins
        fill!(collK_1, 0.0 + 0.0im)
        fill!(collK_2, 0.0 + 0.0im)
        if compute_spectral
            fill!(colls, 0.0 + 0.0im)
        end

        for s in 1:t1, i in 1:N_FIELD, l in 1:N_FIELD
            acc = 0.0 + 0.0im
            for j in 1:N_FIELD
                acc += state.sigmas.data[n, i, j, t1, s] * state.gK.data[n, j, l, s, t2]
            end
            collK_1[i, l] += w1[s] * acc
        end

        for s in 1:t2, i in 1:N_FIELD, l in 1:N_FIELD
            acc = 0.0 + 0.0im
            for j in 1:N_FIELD
                acc += state.sigmaK.data[n, i, j, t1, s] * state.gs.data[n, j, l, s, t2]
            end
            collK_2[i, l] += w2[s] * acc
        end

        if compute_spectral && (t1 != t2)
            for s in 1:t1, i in 1:N_FIELD, l in 1:N_FIELD
                acc = 0.0 + 0.0im
                for j in 1:N_FIELD
                    acc += state.sigmas.data[n, i, j, t1, s] * state.gs.data[n, j, l, s, t2]
                end
                colls[i, l] += w1[s] * acc
            end
            for s in 1:t2, i in 1:N_FIELD, l in 1:N_FIELD
                acc = 0.0 + 0.0im
                for j in 1:N_FIELD
                    acc += state.sigmas.data[n, i, j, t1, s] * state.gs.data[n, j, l, s, t2]
                end
                colls[i, l] -= w2[s] * acc
            end
        end

        for i in 1:N_FIELD, l in 1:N_FIELD
            hfg = 0.0 + 0.0im
            hfs = 0.0 + 0.0im
            for p in 1:N_FIELD, q in 1:N_FIELD
                hfg += K0_MATRIX[i, p] * state.heff[n, p, q, t1] * state.gK.data[n, q, l, t1, t2]
                if compute_spectral
                    hfs += K0_MATRIX[i, p] * state.heff[n, p, q, t1] * state.gs.data[n, q, l, t1, t2]
                end
            end

            kc1 = 0.0 + 0.0im
            kc2 = 0.0 + 0.0im
            kcs = 0.0 + 0.0im
            for p in 1:N_FIELD
                kc1 += K0_MATRIX[i, p] * collK_1[p, l]
                kc2 += K0_MATRIX[i, p] * collK_2[p, l]
                if compute_spectral
                    kcs += K0_MATRIX[i, p] * colls[p, l]
                end
            end

            dgK[n, i, l] = 2im * hfg + im * kc1 - im * kc2
            if compute_spectral
                dgs[n, i, l] = 2im * hfs + im * kcs
            end
        end
    end
    return nothing
end

@inline function _write_rhs_output!(out, idx::Int, src)
    # KadanoffBaym's cache may reuse array objects across RHS slots, so the
    # solver-facing value must be independent even though the work buffer is not.
    out[idx] = copy(src)
    return nothing
end

function _fv!(out, state::SimulationState, ts::Vector{<:Real}, w1, w2, t1::Int, t2::Int; compute_spectral::Bool = true)
    scratch = _scratch(state)
    dgK = scratch.dgK
    dgs = scratch.dgs
    _fv_arrays!(dgK, dgs, state, ts, w1, w2, t1, t2, scratch; compute_spectral = compute_spectral)
    _write_rhs_output!(out, 1, dgK)
    _write_rhs_output!(out, 2, dgs)
    return nothing
end

function _fd!(out, state::SimulationState, ts::Vector{<:Real}, w1, w2, t1::Int, t2::Int)
    if t1 == t2
        _enforce_gs_equal_time_commutator!(state, t1)
    end
    scratch = _scratch(state)
    dgK = scratch.dgK
    dgs = scratch.dgs
    _fv_arrays!(dgK, dgs, state, ts, w1, w2, t1, t2, scratch; compute_spectral = false)

    # Equal-time diagonal update: Keldysh is transpose-symmetric, spectral is fixed by commutation.
    @inbounds for n in axes(dgK, 1), i in 1:N_FIELD, j in i:N_FIELD
        s = dgK[n, i, j] + dgK[n, j, i]
        dgK[n, i, j] = s
        dgK[n, j, i] = s
    end
    fill!(dgs, 0.0 + 0.0im)
    _write_rhs_output!(out, 1, dgK)
    _write_rhs_output!(out, 2, dgs)
    return nothing
end

# -----------------------------------------------------------------------------
# Audits, saving, postprocessing
# -----------------------------------------------------------------------------

function spin_expectation_from_gk(gK_data::Array{ComplexF64, 5})
    n_spins = size(gK_data, 1)
    nt = size(gK_data, 4)
    out = zeros(Float64, n_spins, N_COMP, nt)

    for n in 1:n_spins, t in 1:nt, alpha in 1:N_COMP
        out[n, alpha, t] = real(0.125im * tr(gK_data[n, :, :, t, t] * K_MATRICES[alpha]))
    end
    return out
end

function _max_symmetry_residual_gK(gK_data::Array{ComplexF64, 5})
    n_spins = size(gK_data, 1)
    nt = size(gK_data, 4)
    rmax = 0.0
    for n in 1:n_spins, t1 in 1:nt, t2 in 1:t1
        lhs = @view gK_data[n, :, :, t1, t2]
        rhs = transpose(@view(gK_data[n, :, :, t2, t1]))
        rmax = max(rmax, maximum(abs.(lhs .- rhs)))
    end
    return rmax
end

function _max_symmetry_residual_gs(gs_data::Array{ComplexF64, 5})
    n_spins = size(gs_data, 1)
    nt = size(gs_data, 4)
    rmax = 0.0
    for n in 1:n_spins, t1 in 1:nt, t2 in 1:t1
        lhs = @view gs_data[n, :, :, t1, t2]
        rhs = -transpose(@view(gs_data[n, :, :, t2, t1]))
        rmax = max(rmax, maximum(abs.(lhs .- rhs)))
    end
    return rmax
end

function _max_equal_time_residual_gs(gs_data::Array{ComplexF64, 5}, n_spins::Int)
    nt = size(gs_data, 4)
    target = im .* K0_MATRIX
    rmax = 0.0
    for t in 1:nt
        for n in 1:n_spins
            rmax = max(rmax, maximum(abs.(@view(gs_data[n, :, :, t, t]) .- target)))
        end
    end
    return rmax
end

function _max_kernel_skewhermitian_residual(kernels::KernelBundle)
    ndiss = size(kernels.xiK_table, 1)
    ntau = length(kernels.tau_grid)
    rmax = 0.0

    for d in 1:ndiss, alpha in 1:N_COMP
        for i in 1:ntau
            j = ntau - i + 1
            xplus = kernels.xiK_table[d, alpha, i]
            xminus = kernels.xiK_table[d, alpha, j]
            rmax = max(rmax, abs(xminus + conj(xplus)))
        end
    end
    return rmax
end

function audit_solution(state::SimulationState, ts::Vector{<:Real}; tol::Real = state.params.symmetry_tol)
    gK_data = copy(state.gK.data[:, :, :, 1:length(ts), 1:length(ts)])
    gs_data = copy(state.gs.data[:, :, :, 1:length(ts), 1:length(ts)])

    shape_ok = (size(gK_data, 1) == state.params.n_spins) && (size(gK_data, 2) == N_FIELD) && (size(gK_data, 3) == N_FIELD) &&
               (size(gs_data, 1) == state.params.n_spins) && (size(gs_data, 2) == N_FIELD) && (size(gs_data, 3) == N_FIELD) &&
               (size(gK_data, 4) == length(ts)) && (size(gK_data, 5) == length(ts))

    finite_ok = all(isfinite, real.(gK_data)) && all(isfinite, imag.(gK_data)) &&
                all(isfinite, real.(gs_data)) && all(isfinite, imag.(gs_data))
    observable_finite = all(isfinite, spin_expectation_from_gk(gK_data))

    sym_gK = _max_symmetry_residual_gK(gK_data)
    sym_gs = _max_symmetry_residual_gs(gs_data)
    et_gs = _max_equal_time_residual_gs(gs_data, state.params.n_spins)
    ker_skew = _max_kernel_skewhermitian_residual(state.kernels)

    return (
        shape_ok = shape_ok,
        finite = finite_ok,
        observable_finite = observable_finite,
        max_residual_gK_symmetry = sym_gK,
        max_residual_gs_antisymmetry = sym_gs,
        max_residual_gs_equal_time = et_gs,
        max_residual_kernel_skewhermitian = ker_skew,
        tolerance = tol,
        passed = shape_ok && finite_ok && observable_finite &&
                 (sym_gK <= tol) && (sym_gs <= tol) && (et_gs <= tol) && (ker_skew <= 10tol),
    )
end

function save_raw_output(path::AbstractString, params::OpenSpinsParameters, state::SimulationState, solution, audit)
    nt = length(solution.t)
    gK_full = copy(state.gK.data[:, :, :, 1:nt, 1:nt])
    gs_full = copy(state.gs.data[:, :, :, 1:nt, 1:nt])
    sigmaK_full = copy(state.sigmaK.data[:, :, :, 1:nt, 1:nt])
    sigmas_full = copy(state.sigmas.data[:, :, :, 1:nt, 1:nt])

    payload = (
        metadata = (
            schema_version = 2,
            axis_order = (
                g = "(n, a, b, t, t')",
                sigma = "(n, a, b, t, t')",
                D = "(diss_site, alpha, t, t')",
                Pi = "(diss_site, alpha, t, t')",
                Omega = "(n, alpha, beta, t, t')",
                Mcheck = "(n, n', alpha, beta, t, t')",
            ),
            dissipative_spins = copy(params.dissipative_spins),
            component_labels = COMP_LABELS,
        ),
        params = _parameters_data(params),
        t = copy(solution.t),
        gK = gK_full,
        gs = gs_full,
        sigmaK = sigmaK_full,
        sigmas = sigmas_full,
        dK = copy(state.dK.data[:, :, 1:nt, 1:nt]),
        ds = copy(state.ds.data[:, :, 1:nt, 1:nt]),
        mcheckK = copy(state.mcheckK.data[:, :, :, :, 1:nt, 1:nt]),
        mchecks = copy(state.mchecks.data[:, :, :, :, 1:nt, 1:nt]),
        lambda_bar = copy(state.lambda_bar[:, :, 1:nt]),
        Lambda_bar = copy(state.Lambda_bar[:, :, 1:nt]),
        heff = copy(state.heff[:, :, :, 1:nt]),
        kernels = (
            tau_grid = copy(state.kernels.tau_grid),
            xiK_table = copy(state.kernels.xiK_table),
            xis_table = copy(state.kernels.xis_table),
            has_channel = copy(state.kernels.has_channel),
        ),
        audit = audit,
    )

    open(path, "w") do io
        serialize(io, payload)
    end
    return path
end

function load_raw_output(path::AbstractString)
    open(path, "r") do io
        return deserialize(io)
    end
end

function _serialization_roundtrip_check(path::AbstractString, nt_expected::Int)
    try
        raw = load_raw_output(path)
        return (raw isa NamedTuple) && hasproperty(raw, :t) && (length(raw.t) == nt_expected)
    catch
        return false
    end
end

# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------

"""
Run the open-spin KBE simulation.

Workflow:
1. Build dissipative-spin mapping and precompute Xi^{K/s}(tau) kernels.
2. Build/interpolate kernels once (Integrals.jl + Interpolations.jl).
3. Initialize Green functions and one-time fields.
4. Evolve g^{K/s} with `kbsolve!`.
5. Save raw tensors for separate postprocessing.
"""
function run_simulation(
    params::OpenSpinsParameters;
    initial_state::OpenSpinsInitialState = OpenSpinsInitialState(params.n_spins),
    output_dir::AbstractString = pwd(),
    save_output::Bool = true,
    stop = ts -> false,
)
    size(initial_state.spin_expectation) == (params.n_spins, N_COMP) ||
        error("initial_state.spin_expectation must have shape (n_spins, 3).")

    kernels = precompute_bath_kernels(params)
    state = initialize_state(params, kernels, initial_state)

    # Initialize auxiliary quantities at t=t'=0.
    _update_auxiliaries!(state, [0.0], [0.0], [0.0], 1, 1)

    sol = kbsolve!(
        (out, x...) -> _fv!(out, state, x...),
        (out, x...) -> _fd!(out, state, x...),
        [state.gK, state.gs],
        (0.0, params.tmax);
        callback = (x...) -> _update_auxiliaries!(state, x...),
        atol = params.atol,
        rtol = params.rtol,
        dtini = params.dtini,
        dtmax = params.dtmax,
        stop = stop,
    )
    _trim_point_capacity!(state, length(sol.t))

    audit = audit_solution(state, sol.t; tol = params.symmetry_tol)

    raw_path = ""
    serialization_roundtrip = false
    if save_output
        mkpath(output_dir)
        raw_path = joinpath(output_dir, params.output_basename)
        save_raw_output(raw_path, params, state, sol, audit)
        serialization_roundtrip = _serialization_roundtrip_check(raw_path, length(sol.t))
    end

    audit = (; audit..., serialization_roundtrip = serialization_roundtrip)

    return (
        solution = sol,
        state = state,
        audit = audit,
        raw_output_path = raw_path,
        spin_expectation = spin_expectation_from_gk(state.gK.data[:, :, :, 1:length(sol.t), 1:length(sol.t)]),
    )
end
