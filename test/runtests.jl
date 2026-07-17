using Test

include(joinpath(@__DIR__, "..", "src", "OpenSpins.jl"))

using .OpenSpins

@testset "package load" begin
    @test isdefined(OpenSpins, :run_simulation)
    @test isdefined(OpenSpins, :BathSpectrum)
    @test OpenSpins.component_index(1) == 1
end

@testset "Schwinger-boson constraint" begin
    gK = zeros(ComplexF64, 1, 4, 4, 1, 1)
    for i in 1:4
        gK[1, i, i, 1, 1] = -2im
    end
    raw = (gK = gK,)
    @test schwinger_boson_constraint_from_raw(raw)[1, 1] == 1.0
end

@testset "interaction profiles" begin
    profile = InteractionProfile(
        1.0;
        events = [
            Quench(1.0, 0.0),
            LinearRamp(2.0, 4.0, 1.0),
            SmoothRamp(5.0, 7.0, 0.0),
        ],
    )

    @test profile(0.5) == 1.0
    @test profile(1.0) == 0.0
    @test profile(2.0) == 0.0
    @test profile(3.0) == 0.5
    @test profile(4.5) == 1.0
    @test profile(6.0) == 0.5
    @test profile(7.0) == 0.0

    data = OpenSpins.interaction_profile_data(profile)
    rebuilt = InteractionProfile(data)
    @test all(rebuilt(t) == profile(t) for t in (0.0, 1.0, 2.5, 4.0, 6.25, 8.0))

    @test_throws ErrorException Quench(-1.0, 0.0)
    @test_throws ErrorException LinearRamp(2.0, 2.0, 0.0)
    @test_throws ErrorException InteractionProfile(1.0; events = [LinearRamp(2.0, 4.0, 0.0), Quench(3.0, 1.0)])
end

@testset "profile parameter reconstruction" begin
    spin_profile = InteractionProfile(1.0; events = [Quench(0.5, 0.0)])
    bath_profile = InteractionProfile(0.0; events = [SmoothRamp(0.25, 0.75, 1.0)])
    params = OpenSpinsParameters(
        n_spins = 1,
        tmax = 1.0,
        spin_spin_profile = spin_profile,
        spin_bath_profile = bath_profile,
        dtmax = 0.1,
    )

    data = OpenSpins._parameters_data(params)
    @test data.spin_spin_profile isa NamedTuple
    @test data.spin_bath_profile isa NamedTuple
    @test !hasproperty(data, :spin_spin_scale)
    @test !hasproperty(data, :spin_bath_scale)

    rebuilt = OpenSpinsParameters(data)
    @test rebuilt.dtmax == 0.1
    @test rebuilt.spin_spin_profile(0.5) == 0.0
    @test rebuilt.spin_bath_profile(0.5) == 0.5
end

@testset "two-time exchange vertex dressing" begin
    J = zeros(Float64, 1, 1, 3, 3)
    J[1, 1, 1, 1] = 2.0
    m = zeros(ComplexF64, 1, 1, 3, 3, 2, 2)
    m[1, 1, 1, 1, 1, 2] = 3.0

    bare = OpenSpins._jmj_diag_entry(m, J, 1, 1, 1, 1, 2)
    dressed = OpenSpins._dressed_jmj_diag_entry(m, J, 1, 1, 1, 1, 2, 0.5, 0.25)
    @test bare == 12.0
    @test dressed == 1.5
end

@testset "profiled simulation histories" begin
    J = zeros(Float64, 2, 2, 3, 3)
    J[1, 2, 3, 3] = 0.2
    J[2, 1, 3, 3] = 0.2
    exchange_profile = InteractionProfile(1.0; events = [Quench(0.002, 0.0)])
    params = OpenSpinsParameters(
        n_spins = 2,
        tmax = 0.004,
        J = J,
        spin_spin_profile = exchange_profile,
        dtini = 0.001,
        dtmax = 0.001,
    )

    mktempdir() do output_dir
        result = run_simulation(params; output_dir = output_dir)
        @test result.audit.passed

        off_indices = findall(t -> t >= 0.002, result.solution.t)
        @test !isempty(off_indices)
        @test iszero(maximum(abs, result.state.sigmaK.data[:, :, :, off_indices, :]))
        @test iszero(maximum(abs, result.state.sigmas.data[:, :, :, off_indices, :]))

        raw = load_raw_output(result.raw_output_path)
        @test raw.metadata.schema_version == 2
        @test raw.params isa NamedTuple
        @test raw.params.spin_spin_profile.events[1].kind === :quench
        @test !hasproperty(raw.params, :spin_spin_scale)
        @test !hasproperty(raw.params, :spin_bath_scale)
        @test size(raw.sigmaK) == size(result.state.sigmaK.data)
    end

    bath = BathSpectrum(
        1,
        3;
        gamma = 0.01,
        s = 1.0,
        omega_c = 1.0,
        temperature = 0.5,
        omega_max = 2.0,
        n_omega = 16,
    )
    bath_params = OpenSpinsParameters(
        n_spins = 1,
        tmax = 0.002,
        dissipative_spins = [1],
        baths = [bath],
        spin_bath_profile = InteractionProfile(0.0),
        kernel_ntau = 33,
        dtini = 0.001,
        dtmax = 0.001,
    )
    bath_result = run_simulation(bath_params; save_output = false)
    @test bath_result.audit.passed
    @test iszero(maximum(abs, bath_result.state.dK.data))
    @test iszero(maximum(abs, bath_result.state.ds.data))

    reactivation_profile = InteractionProfile(
        1.0;
        events = [
            Quench(0.001, 0.0),
            Quench(0.002, 1.0),
            Quench(0.004, 0.0),
        ],
    )
    reactivation_params = OpenSpinsParameters(
        n_spins = 1,
        tmax = 0.005,
        dissipative_spins = [1],
        baths = [bath],
        spin_bath_profile = reactivation_profile,
        kernel_ntau = 33,
        dtini = 0.001,
        dtmax = 0.001,
    )
    reactivation_result = run_simulation(reactivation_params; save_output = false)
    reactivation_t = reactivation_result.solution.t
    reactivation_scale = reactivation_profile.(reactivation_t)
    off_indices = findall(iszero, reactivation_scale)
    reactivated_index = findfirst(i -> 0.002 <= reactivation_t[i] < 0.004, eachindex(reactivation_t))

    @test reactivation_result.audit.passed
    @test !isempty(off_indices)
    @test reactivated_index !== nothing
    @test all(
        iszero(maximum(abs, reactivation_result.state.dK.data[:, :, i, j])) &&
        iszero(maximum(abs, reactivation_result.state.ds.data[:, :, i, j])) &&
        iszero(maximum(abs, reactivation_result.state.sigmaK.data[:, :, :, i, j])) &&
        iszero(maximum(abs, reactivation_result.state.sigmas.data[:, :, :, i, j]))
        for i in eachindex(reactivation_t), j in eachindex(reactivation_t)
        if iszero(reactivation_scale[i]) || iszero(reactivation_scale[j])
    )
    @test maximum(abs, reactivation_result.state.piK.data[:, :, off_indices, off_indices]) > 0
    @test maximum(abs, reactivation_result.state.dK.data[:, :, reactivated_index, 1]) > 0
    @test maximum(abs, reactivation_result.state.sigmaK.data[:, :, :, reactivated_index, 1]) > 0
end
