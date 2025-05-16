module Generators_dip

using LinearAlgebra
using SparseArrays

export Generator_dip
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

A `Generator_dip` object should generally not be instantiated directly, but via
[`hamiltonian_dip`](@ref).

The list of `ops` and `amplitudes` are properties of the `Generator_dip`. They
should not be mutated.

# See also

* [`Operator`](@ref) for static generators, which may be obtained from a
  `Generator_dip` via [`evaluate`](@ref).
"""
struct Generator_dip{OT,AT,DT}

    ops::Vector{OT}
    amplitudes::Vector{AT}
    dresses::Vector{DT}
    dresses_derivatives::Union{Matrix{Function}, Nothing}

    function Generator_dip(ops::Vector{OT}, amplitudes::Vector{AT}, 
                            dresses::Vector{DT}; dresses_derivatives::Union{Matrix{Function}, Nothing}=nothing, deriv_warn=false) where {OT,AT,DT}
        if length(dresses) > length(ops)
            error(
                "The number of dresses cannot exceed the number of operators in a Generator_dip"
            )
        end
        if length(amplitudes) < 1
            error("A Generator_dip requires at least one amplitude")
        end
        if isnothing(dresses_derivatives)
            if deriv_warn
                @warn("No dress derivatives provided.\n 
                    If you want to use the dress derivatives, please provide them as a vector of vectors of functions.\n 
                    The structure should be: [[∂d1╱∂a1, ∂d2╱∂a1, ∂d3╱∂a1, ...], [∂d1╱∂a2, ∂d2╱∂a2, ∂d3╱∂a2, ...], ...].\n")
            end
        elseif size(dresses_derivatives, 1) != length(amplitudes)
            error("The number of dresses derivatives must match the number of amplitudes")
        else
            if size(dresses_derivatives, 2) != length(dresses)
                error("The number of dresses derivatives for each amplitude must match the number of dresses")
            end
        end

        # TODO: update documentation over dresses_derivatives
        """
        dresses_derivatives::Matrix{Function}
        A matrix of functions that represent the derivatives of the dresses with respect to the amplitudes.
        The dress derivatives are functions of the amplitudes and the time.

        columns are the amplitudes and rows are the dresses.
        """

        new{OT,AT,DT}(ops, amplitudes, dresses, dresses_derivatives)
    end

end

function Base.show(io::IO, G::Generator_dip{OT,AT,DT}) where {OT,AT,DT}
    print(io, "Generator_dip($(G.ops), $(G.amplitudes), $(G.dresses), $(G.dresses_derivatives))")
end

function Base.summary(io::IO, G::Generator_dip)
    print(io, "Generator_dip with $(length(G.ops)) ops and $(length(G.amplitudes)) amplitudes and $(length(G.dresses)) dresses")
    if !isnothing(G.dresses_derivatives)
        print(io, " and $(size(G.dresses_derivatives)) dress derivatives")
    end
end

function Base.show(io::IO, ::MIME"text/plain", G::Generator_dip{OT,AT,DT}) where {OT,AT,DT}
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
    if !isnothing(G.dresses_derivatives)
        println(io, " dresses_derivatives::Matrix{Function}:")
        for dressd in G.dresses_derivatives
            print(io, "  ")
            show(io, dressd)
            print(io, "\n")
        end
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
hamiltonian_dip(terms...; ampl_vec=[], dres_der=nothing, check=true, deriv_warn=true) = _make_generator_dip(terms...; ampl_vec, dres_der, check, deriv_warn)

function _make_generator_dip(terms...; ampl_vec=[], dres_der=nothing, check=false, deriv_warn=false)
    ops = Any[]
    drift = Any[]
    amplitudes = Any[]
    dresses = Any[]
    if check
        if (length(terms) == 1) && (terms[1] isa Union{Tuple,Vector})
            @warn("Generator_dip terms may not have been properly expanded")
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
    # Check the dresses derivatives if provided
    if !isnothing(dres_der)
        if check
            if !(dres_der isa Vector{Vector{Function}})
                @warn("Dresses derivatives are not a vector of vectors of functions: $(typeof(dre_der))")
            end
        end
        dres_der = hcat(dres_der...)
    end
    ops = [drift..., ops...]  # narrow eltype
    OT = eltype(ops)
    amplitudes = [amplitudes...]  # narrow eltype
    AT = eltype(amplitudes)
    dresses = [dresses...]  # narrow eltype
    DT = eltype(dresses)
    if length(amplitudes) == 0 || length(dresses) == 0
        # No amplitudes or dresses, so we have a static operator
        (length(drift) > 0) || error("Generator_dip has no terms")
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
            if !isnothing(dres_der)
                @warn("Dresses derivatives are not supported for amplitudes of type $AT")
            end
            return Operator(ops, amplitudes)
        else
            return Generator_dip(ops, amplitudes, dresses; 
                dresses_derivatives=dres_der)
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
    # Now that the coeffs of the amplitudes are known, we can evaluate the dresses to find the constants that will multiply each operator
    dress_coeffs = []
    for (i, dres) in enumerate(generator.dresses)
        coeff = dres(coeffs...)
        if coeff isa Number
            push!(dress_coeffs, coeff)
        else
            error(
                "In `evaluate($generator, $args, vals_dict=$vals_dict)`, the dress $i evaluates to $(typeof(coeff)), not a number"
            )
        end
    end
    dress_coeffs = [dress_coeffs...]  # narrow eltype
    return Operator(generator.ops, dress_coeffs)
end


function evaluate!(op::Operator, generator::Generator_dip, args...; vals_dict=IdDict())
    @assert length(op.ops) == length(generator.ops)
    @assert all(O ≡ P for (O, P) in zip(op.ops, generator.ops))
    amplis = []
    for (i, ampl) in enumerate(generator.amplitudes)
        coeff = evaluate(ampl, args...; vals_dict)
        @assert coeff isa Number
        push!(amplis, coeff)
    end
    for (i, dres) in enumerate(generator.dresses)
        coeff = dres(amplis...)
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
    return Generator_dip(ops, amplitudes, generator.dresses)
end



end
