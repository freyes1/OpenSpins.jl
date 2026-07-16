module OpenSpins

function _push_existing_load_path!(path::AbstractString)
    isdir(path) || return false
    abspath(path) in LOAD_PATH || pushfirst!(LOAD_PATH, abspath(path))
    return true
end

for candidate in (
    get(ENV, "KADANOFFBAYM_JL", ""),
    joinpath(@__DIR__, "..", "KadanoffBaym.jl"),
    joinpath(@__DIR__, "..", "..", "KadanoffBaym.jl"),
    joinpath(homedir(), "PersonalCode", "KadanoffBaym.jl"),
)
    isempty(candidate) && continue
    _push_existing_load_path!(candidate) && break
end

include("OpenSpins_impl.jl")
include("Postprocess.jl")

export BathSpectrum
export OpenSpinsParameters
export OpenSpinsInitialState
export run_simulation
export save_raw_output
export load_raw_output
export audit_solution
export spin_expectation_from_gk
export spin_expectation_from_raw
export two_spin_correlator_from_raw
export purity_from_raw
export component_index

end # module OpenSpins
