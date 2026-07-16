# OpenSpins

Self-contained Julia package wrapper for the OpenSpins simulation code.

The package entry point is `src/OpenSpins.jl`, which loads the implementation
body from `src/OpenSpins_impl.jl`.

At load time the wrapper will also look for a local `KadanoffBaym.jl` checkout
via `ENV["KADANOFFBAYM_JL"]`, nearby sibling folders, or
`~/PersonalCode/KadanoffBaym.jl`.
