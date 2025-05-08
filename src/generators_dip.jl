module Generators_dip

using LinearAlgebra
using SparseArrays

export Generator_dip, Operator_dip, ScaledOperator_dip
export hamiltonian_dip
import ..Controls: get_controls, evaluate, evaluate!, substitute
import ..Generators: Operator, ScaledOperator


@doc raw"""A time-dependent generator.

```julia
Generator_dip (ops::Vector{OT}, amplitudes::Vector{AT})
```

produces an object of type `Generator_dip{OT,AT}` that represents

```math
Ĥ(t)= Ĥ_0 + \sum_l a_l(\{ϵ_k(t)\}) \, Ĥ_l\,,
```

where ``Ĥ_l`` are the `ops` and ``ϵ_k(t)`` are the `amplitudes` and `a_l` are the dress functions. 
``Ĥ(t)`` and ``Ĥ_l`` may represent operators in Hilbert space. If the number of `amplitudes` is less than the number of
`ops`, the first `ops` are considered as drift terms (``Ĥ_0``,
respectively subsequent terms with ``a_l ≡ 1``).

A `Generator` object should generally not be instantiated directly, but via
[`hamiltonian_dip`](@ref).

The list of `ops` and `amplitudes` are properties of the `Generator`. They
should not be mutated.

# See also

* [`Operator`](@ref) for static generators, which may be obtained from a
  `Generator` via [`evaluate`](@ref).
"""
struct Generator_dip{OT,AT,DT}

    ops::Vector{OT}
    amplitudes::Vector{AT}
    dresses::Vector{DT}

    function Generator(ops::Vector{OT}, amplitudes::Vector{AT}, dresses::Vector{DT}) where {OT,AT,DT}
        if length(dresses) > length(ops)
            error(
                "The number of dresses cannot exceed the number of operators in a Generator"
            )
        end
        if length(amplitudes) < 1
            error("A Generator requires at least one amplitude")
        end
        new{OT,AT,DT}(ops, amplitudes, dresses)
    end

end

function Base.show(io::IO, G::Generator{OT,AT,DT}) where {OT,AT,DT}
    print(io, "Generator($(G.ops), $(G.amplitudes), $(G.dresses))")
end

function Base.summary(io::IO, G::Generator)
    print(io, "Generator with $(length(G.ops)) ops and $(length(G.amplitudes)) amplitudes and $(length(G.dresses)) dresses")
end

function Base.show(io::IO, ::MIME"text/plain", G::Generator{OT,AT,DT}) where {OT,AT,DT}
    Base.summary(io, G)
    println(io, "\n ops::Vector{$OT}:")
    for op in G.ops
        print(io, "  ")
        show(io, op)
        print(io, "\n")
    end
    println(io, " amplitudes::Vector{$AT}:")
    for ampl in G.amplitudes
        print(io, "  ")
        show(io, ampl)
        print(io, "\n")
    end
    println(io, " dresses::Vector{$DT}:")
    for dress in G.dresses
        print(io, "  ")
        show(io, dress)
        print(io, "\n")
    end
end


"""Initialize a (usually time-dependent) Hamiltonian.

In general,

```julia
H = hamiltonian_dip(terms...; amplvec, check=true)
```

constructs a Hamiltonian based on the given `terms`. Each term must be an
operator, a tuple `(op, dress)` of an operator and a dress function. `amplvec` is a vector of control amplitudes `ampl`. 
Single operators are considered "drift" terms.

In most cases, each control amplitude will simply be a control function or
vector of pulse values. In general, `ampl` can be an arbitrary object that
depends on one or more controls, which must be obtainable via
[`get_controls(ampl)`](@ref get_controls). See
`QuantumPropagators.Interfaces.check_amplitude` for the required
interface.

The `hamiltonian_dip` function will generally return a [`Generator_dip`](@ref)
instance. However, if none of the given terms are time-dependent, it may return
a static operator (e.g., an `AbstractMatrix` or [`Operator`](@ref)):

The `hamiltonian` function may generate warnings if the `terms` are of an
unexpected type or structure.  These can be suppressed with `check=false`.
"""
hamiltonian_dip(terms...; amplvec, check=true) = _make_generator_dip(terms...; amplvec, check)

function _make_generator_dip(terms...; ampl_vec, check=false)
    ops = Any[]
    drift = Any[]
    amplitudes = Any[]
    dresses = Any[]
    if check
        if (length(terms) == 1) && (terms[1] isa Union{Tuple,Vector})
            @warn("Generator terms may not have been properly expanded")
        end
    end
    for term in terms
        if term isa Union{Tuple,Vector} ######################################## Dresses and operators
            length(term) == 2 || error("time-dependent term must be 2-tuple")
            op, dres = term
            if check
                opType = Union{AbstractMatrix}
                if !(op isa opType) && (dres isa opType)
                    @warn("It looks like (op, dress) in term are reversed")
                end
            end
            i = findfirst(a -> a == dres, dresses)
            if isnothing(i)
                push!(ops, op)
                push!(dresses, dres)
            else
                if check
                    try
                        ops[i] = ops[i] + op
                    catch exc
                        @error(
                            "Collected operators are of a disparate type: $(typeof(ops[i])), $(typeof(op)): $exc"
                        )
                        rethrow()
                    end
                else
                    ops[i] = ops[i] + op
                end
            end
        else
            op = term
            if length(drift) == 0
                push!(drift, op)
            else
                if check
                    try
                        drift[1] = drift[1] + op
                    catch exc
                        @error(
                            "Collected drift operators are of a disparate type: $(typeof(drift[1])), $(typeof(op)): $exc"
                        )
                        rethrow()
                    end
                else
                    drift[1] = drift[1] + op
                end
            end
        end
    end # for term in terms
    #Store now the amplitudes
    StdAmplType = Union{Function,Vector}
    for ampl in ampl_vec
        if check
            if !(ampl isa StdAmplType)
                @warn("Amplitude is not a function or vector: $(typeof(ampl))")
            end
        end
        push!(amplitudes, ampl)
    end
    ops = [drift..., ops...]  # narrow eltype
    OT = eltype(ops)
    amplitudes = [amplitudes...]  # narrow eltype
    AT = eltype(amplitudes)
    dresses = [dresses...]  # narrow eltype
    DT = eltype(dresses)
    if length(amplitudes) == 0 || length(dresses) == 0
        # No amplitudes or dresses, so we have a static operator
        (length(drift) > 0) || error("Generator has no terms")
        return drift[1]
    else
        if check
            if !(isconcretetype(OT))
                @warn("Collected operators are not of a concrete type: $OT")
            end
            if AT ≡ Any
                @warn("Collected amplitudes are of disparate types")
            end
        end
        if (AT <: Number)
            return Operator(ops, amplitudes)
        else
            return Generator_dip(ops, amplitudes, dresses)
        end
    end
end

function get_controls(generator::Generator_dip)
    controls = []
    slots_dict = IdDict()  # utilized as Set of controls we've seen
    for (i, ampl) in enumerate(generator.amplitudes)
        for control in get_controls(ampl)
            if control in keys(slots_dict)
                # We've seen this control before, so we just record the slot
                # where it is referenced
                push!(slots_dict[control], i)
            else
                push!(controls, control)
                slots_dict[control] = [i]
            end
        end
    end
    return Tuple(controls)
end


# During evaluation we need to know the dress functions and the amplitudes
# Start here the modification 
function evaluate(dress::Function, ampl_vector::Vector, args...; vals_dict=IdDict())
    if haskey(vals_dict, dress)
        return vals_dict[dress]
    else
        return evaluatedress(dress, ampl_vector, args...)
    end
end

# Evaluate the dress function
# If ampl is a function, we evaluate it with the given arguments
# If ampl is a vector, we evaluate it with the given arguments
# and return the corresponding value
# Pass the values of the amplitudes to the dress function
# The dress function is a function of the amplitudes
function evaluatedress(dress::Function, ampl_vector::Vector, t::Float64)
    numerical_amplitudes = []
    for (i, ampl) in enumerate(ampl_vector)
        if ampl isa Function
            coeff = evaluate(ampl, t)
            if coeff isa Number
                push!(numerical_amplitudes, coeff)
            else
                error(
                    "In `evaluate($dress, $ampl_vector, $t)`, the amplitude $i evaluates to $(typeof(coeff)), not a number"
                )
            end
        elseif ampl isa Vector
            error(
                "In `evaluate($dress, $ampl_vector, $t)`, the amplitude $i is a vector, not a function: $(typeof(ampl))"
            )
        else
            error("Amplitude is not a function or vector: $(typeof(ampl))")
        end
    end

    return dress(numerical_amplitudes...)
end


function evaluatedress(dress::Function, ampl_vector::Vector, tlist::Vector, n::Int64)
    numerical_amplitudes = []
    for (i, ampl) in enumerate(ampl_vector)
        if ampl isa Function
            coeff = evaluate(ampl, tlist, n)
            if coeff isa Number
                push!(numerical_amplitudes, coeff)
            else
                error(
                    "In `evaluate($dress, $ampl_vector, $t)`, the amplitude $i evaluates to $(typeof(coeff)), not a number"
                )
            end
        elseif ampl isa Vector
            coeff = evaluate(ampl, tlist, n)
            if coeff isa Number
                error(
                    "In `evaluate($dress, $ampl_vector, $t)`, the amplitude $i evaluates to $(typeof(ampl)), not a number"
                )
            end
        else
            error("Amplitude is not a function or vector: $(typeof(ampl))")
        end
    end

    return dress(numerical_amplitudes...)
end










# Some functions to evaluate the dress where made yesterday







function evaluate(generator::Generator_dip, args...; vals_dict=IdDict())
    coeffs = []
    for (i, ampl) in enumerate(generator.amplitudes)
        coeff = evaluate(ampl, args...; vals_dict)
        if coeff isa Number
            push!(coeffs, coeff)
        else
            error(
                "In `evaluate($generator, $args, vals_dict=$vals_dict)`, the amplitude $i evaluates to $(typeof(coeff)), not a number"
            )
        end
    end
    coeffs = [coeffs...]  # narrow eltype
    return Operator(generator.ops, coeffs)
end


function evaluate!(op::Operator, generator::Generator_dip, args...; vals_dict=IdDict())
    @assert length(op.ops) == length(generator.ops)
    @assert all(O ≡ P for (O, P) in zip(op.ops, generator.ops))
    for (i, ampl) in enumerate(generator.amplitudes)
        coeff = evaluate(ampl, args...; vals_dict)
        @assert coeff isa Number
        op.coeffs[i] = coeff
    end
    return op
end


function substitute(generator::Generator_dip, replacements)
    if generator ∈ keys(replacements)
        return replacements[generator]
    end
    ops = [substitute(op, replacements) for op in generator.ops]
    amplitudes = [substitute(ampl, replacements) for ampl in generator.amplitudes]
    return Generator(ops, amplitudes)
end

function substitute(operator::Operator, replacements)
    if operator ∈ keys(replacements)
        return replacements[operator]
    end
    ops = [substitute(op, replacements) for op in operator.ops]
    return Operator(ops, operator.coeffs)
end


end
