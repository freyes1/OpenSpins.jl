using Test

include(joinpath(@__DIR__, "..", "src", "OpenSpins.jl"))

using .OpenSpins

@testset "package load" begin
    @test isdefined(OpenSpins, :run_simulation)
    @test isdefined(OpenSpins, :BathSpectrum)
    @test OpenSpins.component_index(1) == 1
end
