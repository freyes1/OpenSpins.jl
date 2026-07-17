# OpenSpins

Self-contained Julia package wrapper for the OpenSpins simulation code.

The package entry point is `src/OpenSpins.jl`, which loads the implementation
body from `src/OpenSpins_impl.jl`.

At load time the wrapper will also look for a local `KadanoffBaym.jl` checkout
via `ENV["KADANOFFBAYM_JL"]`, nearby sibling folders, or
`~/PersonalCode/KadanoffBaym.jl`.

Time-dependent exchange and bath couplings are specified with reusable profiles:

```julia
profile = InteractionProfile(
    1.0;
    events = [
        Quench(2.0, 0.0),
        SmoothRamp(5.0, 6.0, 1.0),
    ],
)

params = OpenSpinsParameters(
    n_spins = 2,
    tmax = 10.0,
    spin_spin_profile = profile,
    dtmax = 0.05,
)
```

Raw output contains the dressed histories used by the solver and plain profile
event data under `raw.params`. Reconstruct a profile with
`InteractionProfile(raw.params.spin_spin_profile)`.
