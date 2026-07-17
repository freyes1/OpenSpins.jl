using OpenSpins
using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const STEM = "bath_quench_sudden_turn_on"
const OUTPUT_DIR = isempty(ARGS) ? joinpath(ROOT, "results", "bath_quench") : abspath(ARGS[1])

mkpath(OUTPUT_DIR)

h = zeros(Float64, 1, 3)
h[1, 3] = 1.0

bath = BathSpectrum(
    1,
    1;
    gamma = 0.1,
    s = 1.0,
    omega_c = 5.0,
    temperature = 0.1,
    omega_max = 25.0,
    n_omega = 256,
)

bath_profile = InteractionProfile(
    0.0;
    events = [
        Quench(7.0, 1.0),
        Quench(14.0, 0.0),
    ],
)

params = OpenSpinsParameters(
    n_spins = 1,
    tmax = 17.0,
    h = h,
    dissipative_spins = [1],
    baths = [bath],
    spin_bath_profile = bath_profile,
    dtini = 0.01,
    dtmax = 0.2,
    atol = 1e-5,
    rtol = 1e-5,
    kernel_ntau = 801,
    output_basename = "$(STEM)_raw.jls",
)

initial_state = OpenSpinsInitialState(reshape([0.5, 0.0, 0.0], 1, 3); spin_length = 0.5)
result = run_simulation(params; initial_state = initial_state, output_dir = OUTPUT_DIR)

raw = load_raw_output(result.raw_output_path)
spin = spin_expectation_from_raw(raw)
spin_norm = sqrt.(sum(abs2, spin; dims = 2))[:, 1, :]
constraint = schwinger_boson_constraint_from_raw(raw)

observables_path = joinpath(OUTPUT_DIR, "$(STEM)_observables.csv")
open(observables_path, "w") do io
    println(io, "t,Sx,Sy,Sz,spin_norm,boson_constraint,bath_scale")
    for i in eachindex(raw.t)
        @printf(
            io,
            "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
            raw.t[i],
            spin[1, 1, i],
            spin[1, 2, i],
            spin[1, 3, i],
            spin_norm[1, i],
            constraint[1, i],
            bath_profile(raw.t[i]),
        )
    end
end

println("raw_output=$(result.raw_output_path)")
println("observables=$(observables_path)")
println("audit_passed=$(result.audit.passed)")
println("n_time_points=$(length(raw.t))")
println("final_spin=$(spin[1, :, end])")
println("final_constraint=$(constraint[1, end])")
