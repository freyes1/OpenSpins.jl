using OpenSpins
using Printf

length(ARGS) >= 1 || error(
    "Usage: julia --project=. examples/heisenberg_chain_interaction_quench.jl sudden|linear|smooth",
)
const TURN_ON_KIND = Symbol(lowercase(ARGS[1]))
TURN_ON_KIND in (:sudden, :linear, :smooth) || error("Turn-on kind must be sudden, linear, or smooth.")

const ROOT = normpath(joinpath(@__DIR__, ".."))
const STEM = "interaction_quench_$(TURN_ON_KIND)_turn_on"
const OUTPUT_DIR = joinpath(ROOT, "results", "interaction_quench")
const N_SPINS = 4

mkpath(OUTPUT_DIR)

h = zeros(Float64, N_SPINS, 3)
h[:, 1] .= 1.0

J = zeros(Float64, N_SPINS, N_SPINS, 3, 3)
for (n, np) in ((1, 2), (2, 3), (3, 4), (4, 1))
    for alpha in 1:3
        J[n, np, alpha, alpha] = 1.0
        J[np, n, alpha, alpha] = 1.0
    end
end

turn_on = if TURN_ON_KIND === :sudden
    Quench(7.0, 1.0)
elseif TURN_ON_KIND === :linear
    LinearRamp(7.0, 8.0, 1.0)
else
    SmoothRamp(7.0, 8.0, 1.0)
end
interaction_profile = InteractionProfile(0.0; events = [turn_on, Quench(14.0, 0.0)])

params = OpenSpinsParameters(
    n_spins = N_SPINS,
    tmax = 17.0,
    h = h,
    J = J,
    spin_spin_profile = interaction_profile,
    dtini = 0.01,
    dtmax = 0.2,
    output_basename = "$(STEM)_raw.jls",
)

initial_spins = zeros(Float64, N_SPINS, 3)
initial_spins[:, 3] .= [0.5, -0.5, 0.5, -0.5]
initial_state = OpenSpinsInitialState(initial_spins; spin_length = 0.5)
result = run_simulation(params; initial_state = initial_state, output_dir = OUTPUT_DIR)

raw = load_raw_output(result.raw_output_path)
spin = spin_expectation_from_raw(raw)
spin_norm = sqrt.(sum(abs2, spin; dims = 2))[:, 1, :]
constraint = schwinger_boson_constraint_from_raw(raw)
staggered = zeros(Float64, 3, length(raw.t))
uniform = zeros(Float64, 3, length(raw.t))
for i in eachindex(raw.t), alpha in 1:3
    staggered[alpha, i] = sum((-1)^(n - 1) * spin[n, alpha, i] for n in 1:N_SPINS) / N_SPINS
    uniform[alpha, i] = sum(spin[n, alpha, i] for n in 1:N_SPINS) / N_SPINS
end

observables_path = joinpath(OUTPUT_DIR, "$(STEM)_observables.csv")
open(observables_path, "w") do io
    header = ["t"]
    append!(header, ["S$(n)$(axis)" for n in 1:N_SPINS for axis in ("x", "y", "z")])
    append!(header, ["spin_norm$(n)" for n in 1:N_SPINS])
    append!(header, ["boson_constraint$(n)" for n in 1:N_SPINS])
    append!(header, ["staggered_$(axis)" for axis in ("x", "y", "z")])
    append!(header, ["uniform_$(axis)" for axis in ("x", "y", "z")])
    push!(header, "interaction_scale")
    println(io, join(header, ","))

    for i in eachindex(raw.t)
        row = Float64[raw.t[i]]
        append!(row, [spin[n, alpha, i] for n in 1:N_SPINS for alpha in 1:3])
        append!(row, spin_norm[:, i])
        append!(row, constraint[:, i])
        append!(row, staggered[:, i])
        append!(row, uniform[:, i])
        push!(row, interaction_profile(raw.t[i]))
        println(io, join((@sprintf("%.12g", value) for value in row), ","))
    end
end

println("turn_on_kind=$(TURN_ON_KIND)")
println("raw_output=$(result.raw_output_path)")
println("observables=$(observables_path)")
println("audit_passed=$(result.audit.passed)")
println("n_time_points=$(length(raw.t))")
println("final_staggered=$(staggered[:, end])")
println("max_constraint_error=$(maximum(abs.(constraint .- 1.0)))")
