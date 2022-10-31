module Generators

using LinearAlgebra
using SparseArrays

export Generator, Operator, ScaledOperator
export hamiltonian, liouvillian
import ..Controls: getcontrols, evalcontrols, evalcontrols!, substitute_controls


@doc raw"""A time-dependent generator.

```julia
Generator(ops::Vector{OT}, amplitudes::Vector{AT})
```

produces an object of type `Generator{OT,AT}` that represents

```math
Ĥ(t)= Ĥ_0 + \sum_l a_l(\{ϵ_{l'}(t)\}, t) \, Ĥ_l\,,
```

where ``Ĥ_l`` are the `ops` and ``a_l(t)`` are the `amplitudes`. ``Ĥ(t)`` and
``Ĥ_l`` may represent operators in Hilbert space or super-operators in
Liouville space. If the number of `amplitudes` is less than the number of
`ops`, the first `ops` are considered as drift terms (``Ĥ_0``,
respectively subsequent terms with ``a_l ≡ 1``).
At least one time-dependent amplitude is required. Each amplitude may depend on
one or more control functions ``ϵ_{l'}(t)``, although most typically ``a_l(t) ≡
ϵ_l(t)``, that is, the `amplitudes` are simply a vector of the controls. See
[`hamiltonian`](@ref) for details.

A `Generator` object should generally not be instantiated directly, but via
[`hamiltonian`](@ref) or [`liouvillian`](@ref).

The list of `ops` and `amplitudes` are properties of the `Generator`. They
should not be mutated.

# See also

* [`Operator`](@ref) for static generators, which may be obtained from a
  `Generator` via [`evalcontrols`](@ref evalcontrols).
"""
struct Generator{OT,AT}

    ops::Vector{OT}
    amplitudes::Vector{AT}

    function Generator(ops::Vector{OT}, amplitudes::Vector{AT}) where {OT,AT}
        if length(amplitudes) > length(ops)
            error(
                "The number of amplitudes cannot exceed the number of operators in a Generator"
            )
        end
        if length(amplitudes) < 1
            error("A Generator requires at least one amplitude")
        end
        new{OT,AT}(ops, amplitudes)
    end

end

function Base.show(io::IO, G::Generator{OT,AT}) where {OT,AT}
    print(
        io,
        "Generator{$OT, $AT}(<$(length(G.ops)) ops>, <$(length(G.amplitudes)) amplitudes>)"
    )
end


@doc raw"""A static operator in Hilbert or Liouville space.

```julia
Operator(ops::Vector{OT}, coeffs::Vector{CT})
```

produces an object of type `Operator{OT,CT}` that encapsulates the "lazy" sum

```math
Ĥ = \sum_l c_l Ĥ_l\,,
```

where ``Ĥ_l`` are the `ops` and ``c_l`` are the `coeffs`, which each must be a
constant `Number`. If the number of coefficients is less than the
number of operators, the first `ops` are considered to have ``c_l = 1``.

An `Operator` object would generally not be instantiated directly, but be
obtained from a [`Generator`](@ref) via [`evalcontrols`](@ref evalcontrols).
"""
struct Operator{OT,CT<:Number}

    ops::Vector{OT}
    coeffs::Vector{CT}

    function Operator(ops::Vector{OT}, coeffs::Vector{CT}) where {OT,CT}
        if length(coeffs) > length(ops)
            error(
                "The number of coefficients cannot exceed the number of operators in an Operator"
            )
        end
        new{OT,CT}(ops, coeffs)
    end

end


function Base.show(io::IO, O::Operator{OT,CT}) where {OT,CT}
    print(io, "Operator{$OT, $CT}(<$(length(O.ops)) ops>, <$(length(O.coeffs)) coeffs>)")
end


function Base.Array{T}(O::Operator) where {T}
    A = Array{T}(O.ops[1])
    drift_offset = length(O.ops) - length(O.coeffs)
    if drift_offset == 0
        lmul!(O.coeffs[1], A)
    end
    N = length(O.ops)
    for i = 2:N
        X = Array{T}(O.ops[i])
        if i > drift_offset
            a = O.coeffs[i-drift_offset]
        else
            a = true
        end
        axpy!(a, X, A)
    end
    return A
end

Base.Array(O::Operator) = Array{ComplexF64}(O)


function Base.copy(O::Operator)
    return Operator([copy(op) for op in O.ops], copy(O.coeffs))
end


function Base.copyto!(tgt::Operator, src::Operator)
    for (i, op) in enumerate(src.ops)
        copyto!(tgt.ops[i], op)
    end
    copyto!(tgt.coeffs, src.coeffs)
end

Base.size(O::Operator) = size(O.ops[1])


function LinearAlgebra.ishermitian(O::Operator{OT,CT}) where {OT,CT}
    return all(ishermitian(op) for op in O.ops) && all(isreal(c) for c in O.coeffs)
end



"""A static operator with a scalar pre-factor.

```julia
op = ScaledOperator(α, Ĥ)
```

represents the "lazy" product ``α Ĥ`` where ``Ĥ`` is an operator (typically an
[`Operator`](@ref) instance) and ``α`` is a scalar.
"""
struct ScaledOperator{CT<:Number,OT}
    coeff::CT
    operator::OT

    function ScaledOperator(coeff::CT, operator::OT) where {CT,OT}
        if coeff == 1.0
            return operator
        else
            return new{CT,OT}(coeff, operator)
        end
    end
end


function Base.show(io::IO, O::ScaledOperator{CT,OT}) where {CT,OT}
    print(io, "ScaledOperator{$CT,$(nameof(OT))}($(O.coeff), …)")
end


function Base.Array{T}(O::ScaledOperator{CT,Operator{OOT,OCT}}) where {T,CT<:Number,OOT,OCT}
    A = Array{T}(O.operator.ops[1])
    drift_offset = length(O.operator.ops) - length(O.operator.coeffs)
    a = O.coeff
    (drift_offset == 0) && (a *= O.operator.coeffs[1])
    lmul!(a, A)
    N = length(O.operator.ops)
    for i = 2:N
        X = Array{T}(O.operator.ops[i])
        a = O.coeff
        (i > drift_offset) && (a *= O.operator.coeffs[i-drift_offset])
        axpy!(a, X, A)
    end
    return A
end

# fallback (less efficient, but doesn't assume Operator-OT)
Base.Array{T}(O::ScaledOperator{CT,OT}) where {T,CT,OT} = O.coeff * Array{T}(O.operator)
Base.Array(O::ScaledOperator) = Array{ComplexF64}(O)


Base.size(O::ScaledOperator) = size(O.operator)

LinearAlgebra.ishermitian(O::ScaledOperator) = (isreal(O.coeff) && ishermitian(O.operator))


"""Initialize a (usually time-dependent) Hamiltonian.

The most common usage is, e.g.,

```jldoctest hamiltonian
using QuantumPropagators

H₀ = ComplexF64[0 0; 0 1];
H₁ = ComplexF64[0 1; 1 0];
ϵ₁(t) = 1.0;

hamiltonian(H₀, (H₀, ϵ₁))

# output

Generator{Matrix{ComplexF64}, typeof(ϵ₁)}(<2 ops>, <1 amplitudes>)
```

In general,

```julia
H = hamiltonian(terms...; check=true)
```

constructs a Hamiltonian based on the given `terms`. Each term must be an
operator or a tuple `(op, ampl)` of an operator and a control amplitude. Single
operators are considered "drift" terms.

In most cases, each control amplitude will simply be a control function or
vector of pulse values. In general, `ampl` can be an arbitrary object that
depends on one or more controls, which must be obtainable via
[`getcontrols(ampl)`](@ref getcontrols).

The `hamiltonian` function will generally return a [`Generator`](@ref)
instance. However, if none of the given terms are time-dependent, it may return
a static operator (e.g., an `AbstractMatrix` or [`Operator`](@ref)):

```jldoctest hamiltonian
hamiltonian(H₀)
# output
2×2 Matrix{ComplexF64}:
 0.0+0.0im  0.0+0.0im
 0.0+0.0im  1.0+0.0im
```

```jldoctest hamiltonian
hamiltonian(H₀, (H₁, 2.0))
# output
Operator{Matrix{ComplexF64}, Float64}(<2 ops>, <1 coeffs>)
```

The `hamiltonian` function may generate warnings if the `terms` are of an
unexpected type or structure as well was warnings or errors for any `ampl` that
does not implement the interface required for control amplitudes.  These can be
suppressed with `check=false`.
"""
hamiltonian(terms...; check=true) = _make_generator(terms...; check)

function _make_generator(terms...; check=false)
    ops = Any[]
    drift = Any[]
    amplitudes = Any[]
    if check
        if (length(terms) == 1) && (terms[1] isa Union{Tuple,Vector})
            @warn("Generator terms may not have been properly expanded")
        end
    end
    for term in terms
        if term isa Union{Tuple,Vector}
            length(term) == 2 || error("time-dependent term must be 2-tuple")
            op, ampl = term
            if check
                StdAmplType = Union{Function,Vector}
                if (op isa StdAmplType) && !(ampl isa StdAmplType)
                    @warn("It looks like (op, ampl) in term are reversed")
                end
            end
            i = findfirst(a -> a == ampl, amplitudes)
            if isnothing(i)
                push!(ops, op)
                push!(amplitudes, ampl)
            else
                try
                    ops[i] = ops[i] + op
                catch
                    @error(
                        "Collected operators are of a disparate type: $(typeof(ops[i])), $(typeof(op))"
                    )
                    rethrow()
                end
            end
        else
            op = term
            if length(drift) == 0
                push!(drift, op)
            else
                try
                    drift[1] = drift[1] + op
                catch
                    @error(
                        "Collected drift operators are of a disparate type: $(typeof(drift[1])), $(typeof(op))"
                    )
                    rethrow()
                end
            end
        end
    end
    ops = [drift..., ops...]  # narrow eltype
    OT = eltype(ops)
    amplitudes = [amplitudes...]  # narrow eltype
    AT = eltype(amplitudes)
    if length(amplitudes) == 0
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
            if check
                for (i, ampl) in enumerate(amplitudes)
                    if !check_amplitude(ampl; throw_error=false)
                        @warn("Collected amplitude #$i is invalid")
                    end
                end
            end
            return Generator(ops, amplitudes)
        end
    end
end



function ham_to_superop(H::AbstractSparseMatrix; convention)
    # See https://arxiv.org/abs/1312.0111, Appendix B.2
    ⊗(A, B) = kron(A, B)
    𝟙 = SparseMatrixCSC{ComplexF64,Int64}(sparse(I, size(H)[1], size(H)[2]))
    H_T = sparse(transpose(H))
    L = sparse(𝟙 ⊗ H - H_T ⊗ 𝟙)
    if convention == :TDSE
        return L
    elseif convention == :LvN
        return 1im * L
    else
        throw(ArgumentError("convention must be :TDSE or :LvN"))
    end
end

function ham_to_superop(H::AbstractMatrix; convention)
    return ham_to_superop(sparse(H); convention=convention)
end


function lindblad_to_superop(A::AbstractSparseMatrix; convention)
    # See https://arxiv.org/abs/1312.0111, Appendix B.2
    ⊗(A, B) = kron(A, B)
    A⁺ = sparse(A')
    A⁺ᵀ = sparse(transpose(A⁺))
    A⁺_A = sparse(A⁺ * A)
    A⁺_A_ᵀ = sparse(transpose(A⁺_A))
    𝟙 = SparseMatrixCSC{ComplexF64,Int64}(sparse(I, size(A)[1], size(A)[2]))
    D = sparse(A⁺ᵀ ⊗ A - (𝟙 ⊗ A⁺_A) / 2 - (A⁺_A_ᵀ ⊗ 𝟙) / 2)
    if convention == :TDSE
        return 1im * D
    elseif convention == :LvN
        return D
    else
        throw(ArgumentError("convention must be :TDSE or :LvN"))
    end
end

function lindblad_to_superop(A::AbstractMatrix; convention)
    return lindblad_to_superop(sparse(A); convention=convention)
end


function dissipator(c_ops; convention)
    N = size(c_ops[1])[1]
    @assert N == size(c_ops[1])[2]
    D = spzeros(ComplexF64, N^2, N^2)
    for A in c_ops
        D += lindblad_to_superop(A; convention=convention)
    end
    return (D,)
end


nhilbert(H::AbstractMatrix) = size(H)[1]
nhilbert(H::Tuple{HT,ET}) where {HT<:AbstractMatrix,ET} = size(H[1])[1]


@doc raw"""Construct a Liouvillian [`Generator`](@ref).

```julia
ℒ = liouvillian(Ĥ, c_ops=(); convention=:LvN, check=true)
```

calculates the sparse Liouvillian super-operator `ℒ` from the Hamiltonian `Ĥ`
and a list `c_ops` of Lindblad operators.

With `convention=:LvN`, applying the resulting `ℒ` to a vectorized density
matrix `ρ⃗` calculates ``\frac{d}{dt} \vec{\rho}(t) = ℒ \vec{\rho}(t)``
equivalent to the Liouville-von-Neumann equation for the density matrix ``ρ̂``,

```math
\frac{d}{dt} ρ̂(t)
= -i [Ĥ, ρ̂(t)] + \sum_k\left(
    Â_k ρ̂ Â_k^\dagger
    - \frac{1}{2} A_k^\dagger Â_k ρ̂
    - \frac{1}{2} ρ̂ Â_k^\dagger Â_k
  \right)\,,
```

where the Lindblad operators ``Â_k`` are the elements of `c_ops`.

The Hamiltonian ``Ĥ`` will generally be time-dependent. For example, it may be
a [`Generator`](@ref) as returned by [`hamiltonian`](@ref). For example, for a
Hamiltonian with the terms `(Ĥ₀, (Ĥ₁, ϵ₁), (Ĥ₂, ϵ₂))`, where `Ĥ₀`, `Ĥ₁`, `Ĥ₂`
are matrices, and `ϵ₁` and `ϵ₂` are functions of time, the resulting `ℒ` will
be a [`Generator`](@ref) corresponding to terms `(ℒ₀, (ℒ₁, ϵ₁), (ℒ₂, ϵ₂))`,
where the initial terms is the superoperator `ℒ₀` for the static component of
the Liouvillian, i.e., the commutator with the drift Hamiltonian `Ĥ₀`, plus the
dissipator (sum over ``k``), as a sparse matrix. Time-dependent Lindblad
operators are not currently supported. The remaining elements are tuples `(ℒ₁,
ϵ₁)` and `(ℒ₂, ϵ₂)` corresponding to the commutators with the two control
Hamiltonians, where `ℒ₁` and `ℒ₂` again are sparse matrices.

If ``Ĥ`` is not time-dependent, the resulting `ℒ` will likewise be a static
operator. Passing `H=nothing` with non-empty `c_ops` initializes a pure
dissipator.

With `convention=:TDSE`, the Liouvillian will be constructed for the equation
of motion ``i \hbar \frac{d}{dt} \vec{\rho}(t) = ℒ \vec{\rho}(t)`` to match
exactly the form of the time-dependent Schrödinger equation. While this
notation is not standard in the literature of open quantum systems, it has the
benefit that the resulting `ℒ` can be used in a numerical propagator for a
(non-Hermitian) Schrödinger equation without any change. Thus, for numerical
applications, `convention=:TDSE` is generally preferred. The returned `ℒ`
between the two conventions differs only by a factor of ``i``, since we
generally assume ``\hbar=1``.

The `convention` keyword argument is mandatory, to force a conscious choice.

See [Goerz et. al. "Optimal control theory for a unitary operation under
dissipative evolution", arXiv 1312.0111v2, Appendix
B.2](https://arxiv.org/abs/1312.0111v2) for the explicit construction of the
Liouvillian superoperator as a sparse matrix.

Passing `check=false`, suppresses warnings and errors about unexpected types or
the structure of the arguments, cf. [`hamiltonian`](@ref).
"""
function liouvillian(H::Tuple, c_ops=(); kwargs...)
    check = get(kwargs, :check, true)
    return liouvillian(_make_generator(H...; check), c_ops; kwargs...)
end

function liouvillian(H::Generator, c_ops=(); convention, check=true)
    terms = []
    if length(c_ops) > 0
        append!(terms, dissipator(c_ops; convention))
    end
    drift_offset = length(H.ops) - length(H.amplitudes)
    for (i, op) in enumerate(H.ops)
        if i <= drift_offset
            term = ham_to_superop(op; convention)
            push!(terms, term)
        else
            ampl = H.amplitudes[i-drift_offset]
            term = (ham_to_superop(op; convention), ampl)
            push!(terms, term)
        end
    end
    return _make_generator(terms...; check)
end

function liouvillian(H::AbstractMatrix, c_ops=(); convention, check=true)
    L0 = ham_to_superop(H; convention)
    terms = Any[L0,]
    if length(c_ops) > 0
        append!(terms, dissipator(c_ops; convention))
    end
    return _make_generator(terms...; check)
end

function liouvillian(H::Nothing, c_ops=(); convention, check=true)
    if length(c_ops) > 0
        terms = dissipator(c_ops; convention=convention)
        return _make_generator(terms...; check)
    else
        error("Empty Liouvillian, must give at least one of `H` or `c_ops`")
    end
end


function LinearAlgebra.mul!(C, A::Operator, B, α, β)
    drift_offset = length(A.ops) - length(A.coeffs)
    c = α
    (drift_offset == 0) && (c *= A.coeffs[1])
    mul!(C, A.ops[1], B, c, β)
    for i = 2:length(A.ops)
        c = α
        (i > drift_offset) && (c *= A.coeffs[i-drift_offset])
        mul!(C, A.ops[i], B, c, true)
    end
    return C
end


function LinearAlgebra.dot(x, A::Operator, y)
    drift_offset = length(A.ops) - length(A.coeffs)
    result::ComplexF64 = 0
    for i = 1:length(A.ops)
        if i > drift_offset
            c = A.coeffs[i-drift_offset]
            result += c * dot(x, A.ops[i], y)
        else
            result += dot(x, A.ops[i], y)
        end
    end
    return result
end


function Base.:*(α::Number, O::Operator)
    return ScaledOperator(α, O)
end

Base.:*(O::Operator, α::Number) = α * O

Base.convert(::Type{MT}, O::Operator) where {MT<:Matrix} = convert(MT, Array(O))
Base.convert(::Type{MT}, O::ScaledOperator) where {MT<:Matrix} = convert(MT, Array(O))


function Base.:*(α::Number, O::ScaledOperator)
    return ScaledOperator(α * O.coeff, O.operator)
end

Base.:*(O::ScaledOperator, α::Number) = α * O


function LinearAlgebra.mul!(C, A::ScaledOperator, B, α, β)
    return mul!(C, A.operator, B, A.coeff * α, β)
end


function LinearAlgebra.dot(x, A::ScaledOperator, y)
    return A.coeff * dot(x, A.operator, y)
end


function getcontrols(generator::Generator)
    controls = []
    slots_dict = IdDict()  # utilized as Set of controls we've seen
    for (i, ampl) in enumerate(generator.amplitudes)
        for control in getcontrols(ampl)
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


function getcontrols(generator::Tuple)
    return getcontrols(_make_generator(generator...))
end


getcontrols(operator::Operator) = Tuple([])

function evalcontrols(generator::Generator, vals_dict::AbstractDict, args...)
    coeffs = []
    for ampl in generator.amplitudes
        push!(coeffs, evalcontrols(ampl, vals_dict, args...))
    end
    coeffs = [coeffs...]  # narrow eltype
    return Operator(generator.ops, coeffs)
end


function evalcontrols!(op::Operator, generator::Generator, vals_dict::AbstractDict, args...)
    @assert length(op.ops) == length(generator.ops)
    @assert all(O ≡ P for (O, P) in zip(op.ops, generator.ops))
    for (i, ampl) in enumerate(generator.amplitudes)
        op.coeffs[i] = evalcontrols(ampl, vals_dict, args...)
    end
    return op
end

evalcontrols!(op1::T, op2::T, _...) where {T<:AbstractMatrix} = op1
evalcontrols(operator::Operator, _...) = operator
evalcontrols!(op1::T, op2::T, _...) where {T<:Operator} = op1


"""
```julia
ampl = substitute_controls(ampl, controls_map)
```

returns a new amplitude by replacing the original controls that the amplitude
might depend on by the controls in `controls_map`.

Note that for "trivial" amplitudes (where the amplitude is identical to the
control), this simply looks up the control in `controls_map`.
"""
substitute_controls(ampl::Function, controls_map) = get(controls_map, ampl, ampl)
substitute_controls(ampl::Vector, controls_map) = get(controls_map, ampl, ampl)


function substitute_controls(generator::Generator, controls_map)
    amplitudes = [substitute_controls(ampl, controls_map) for ampl in generator.amplitudes]
    return Generator(generator.ops, amplitudes)
end

substitute_controls(operator::Operator, controls_map) = operator


"""Run a check on the given `generator` relative to `state`.

```julia
check_generator(generator, state; throw_error=true)
```

performs a thorough check that all required methods are defined for the type of
the given `generator` under the assumption that `generator` describes the
dynamics of the given `state`.

Returns `true` if the `generator` type passes all checks. Otherwise, throws an
error if `throw_error=true` (default) or return `false`.

Use this to check a custom type for a dynamic generator.
"""
function check_generator(generator, state, throw_error=true)

    GT = typeof(generator)

    local control, controls, operator, μ

    # getcontrols
    if hasmethod(getcontrols, (GT,))
        try
            controls = getcontrols(generator)
        catch exc
            @error("getcontrols(::$GT) returns an invalid result $exc")
            throw_error ? rethrow() : return false
        end
    else
        @error("getcontrols(::$GT) is not implemented")
        throw_error ? error("Invalid generator") : return false
    end
    # TODO: check controls (must be discretizable) etc.

    errors = String[]

    tlist = [0.0, 1.0]
    vals_dict = IdDict(ϵ => 1.0 for ϵ in controls)

    # evalcontrols
    if hasmethod(evalcontrols, (GT, typeof(vals_dict), typeof(tlist), Int64))
        try
            operator = evalcontrols(generator, vals_dict, tlist, 1)
        catch exc
            @error("evalcontrols(::$GT, …) is invalid: $exc")
            throw_error ? rethrow() : return false
        end
        try
            check_operator(operator, state)
        catch exc
            push!(errors, "evalcontrols does not return a valid operator: $exc")
        end
    else
        @error("evalcontrols(::$GT, vals_dict, tlist, n) is not implemented")
        throw_error ? error("Invalid generator") : return false
    end

    # evalcontrols!
    if hasmethod(
        evalcontrols!,
        (typeof(operator), GT, typeof(vals_dict), typeof(tlist), Int64)
    )
        try
            evalcontrols!(operator, generator, vals_dict, tlist, 1)
        catch exc
            push!(errors, "evalcontrols!(…, ::$GT, …) is invalid: $exc")
        end
    else
        push!(errors, "evalcontrols!(op, ::$GT, vals_dict, tlist, n) is not implemented")
    end

    # substitute_controls
    controls_map = IdDict(ϵ => ϵ for ϵ in controls)
    if hasmethod(substitute_controls, (GT, typeof(controls_map)))
        try
            substitute_controls(generator, controls_map)
        catch exc
            push!(errors, "substitute_controls(::$GT, …) is invalid: $exc")
        end
    else
        push!(errors, "substitute_controls(::$GT, …) is not implemented")
    end

    # getcontrolderiv
    #=
    if hasmethod(getcontrolderiv, (GT, eltype(controls)))
        for control in controls
            try
                μ = getcontrolderiv(generator, control)
                if hasmethod(
                    evalcontrols,
                    (typeof(μ), typeof(vals_dict), typeof(tlist), Int64)
                )
                    μ_op = evalcontrols(μ, vals_dict, tlist, 1)
                    check_operator(μ_op, state)
                else
                    push!(
                        errors,
                        "getcontrolderiv(::$GT, …) does not return a proper generator"
                    )
                end
            catch exc
                push!(errors, "getcontrolderiv(::$GT, …) is invalid: $exc")
            end
        end
        v = rand()
        dummy_control = t -> v
        try
            μ = getcontrolderiv(generator, dummy_control)
            if !isnothing(μ)
                push!(
                    errors,
                    "getcontrolderiv(::$GT, …) does not return `nothing` for a control that the generator does not depend on"
                )
            end
        catch exc
            push!(
                errors,
                "getcontrolderiv(::$GT, …)) is invalid for a control that the generator does not depend on: $exc"
            )
        end
    else
        push!(errors, "getcontrolderiv(::$GT, …) is not implemented")
    end
    =#

    # getcontrolderivs has a default implementation, so we don't need to check
    # it

    if length(errors) > 0
        for error in errors
            @error(error)
        end
        throw_error ? error("Invalid generator") : return false
    end

    return true

end



"""Run a check on the given `operator` relative to `state`.

```julia
check_operator(operator, state; throw_error=true)
```

performs a thorough check that all required methods are defined for the type of
the given `operator`. Most importantly, check that `mul!` is implemented to
multiply the `operator` to the given `state`. Note that `copy(state)` must be
implemented.

Return `true` if the `operator` passes all checks. Otherwise, throws an
error if `throw_error=true` (default) or return `false`.

Use this to check a custom type for an operator.
"""
function check_operator(operator, state; throw_error=true)

    OT = typeof(operator)
    ST = typeof(state)
    local state2
    errors = String[]

    # copy/copyto! (warn)
    if hasmethod(copy, (OT,))
        try
            op2 = copy(operator)
            if hasmethod(copyto!, (typeof(op2), OT,))
                try
                    copyto!(op2, operator)
                catch exc
                    push!(errors, "copyto!(…, ::$OT) is invalid: $exc")
                end
            else
                @warn("copyto!(…, ::$OT) is not implemented")
            end
        catch exc
            push!(errors, "copy(::$OT))) is invalid: $exc")
        end
    else
        @warn("copy(::$OT) is not implemented")
    end

    # mul!
    if hasmethod(mul!, (ST, OT, ST, Float64, Float64))
        state2 = copy(state)
        try
            mul!(state2, operator, state, rand(), rand())
        catch exc
            push!(errors, "mul!(…, ::$OT, …) is invalid: $exc")
        end
    else
        push!(errors, "mul!(…, ::$OT, …) is not implemented")
    end

    if length(errors) > 0
        for error in errors
            @error(error)
        end
        throw_error ? error("Invalid operator") : return false
    end

    return true

end


"""Run a check on the given control amplitude.

```julia
check_amplitude(ampl; throw_error=true)
```

checks that `ampl` fulfills the requirements of a control amplitude in a
[`Generator`](@ref).

Return `true` if the `ampl` passes all checks. Otherwise, throws an
error if `throw_error=true` (default) or return `false`.

Use this to check a custom type for an amplitude.
"""
function check_amplitude(ampl; throw_error=true)

    local control, controls, val

    AT = typeof(ampl)

    errors = String[]

    # getcontrols
    if hasmethod(getcontrols, (AT,))
        try
            controls = getcontrols(ampl)
        catch exc
            @error("getcontrols(::$AT) for amplitude returns an invalid result $exc")
            throw_error ? rethrow() : return false
        end
    else
        @error("getcontrols(::$AT) for amplitude is not implemented")
        throw_error ? error("Invalid control amplitude") : return false
    end

    # substitute_controls
    controls_map = IdDict(ϵ => ϵ for ϵ in controls)
    if hasmethod(substitute_controls, (AT, typeof(controls_map)))
        try
            substitute_controls(ampl, controls_map)
        catch exc
            push!(errors, "substitute_controls(::$AT, …) for amplitude is invalid: $exc")
        end
    else
        push!(errors, "substitute_controls(::$AT, …) for amplitude is not implemented")
    end

    vals_dict = IdDict(ϵ => 1.0 for ϵ in controls)

    # evalcontrols
    if hasmethod(evalcontrols, (AT, typeof(vals_dict), Vector{Float64}, Int64))
        tlist = [0.0, 1.0]
        try
            val = evalcontrols(ampl, vals_dict, tlist, 1)
        catch exc
            @error("evalcontrols(::$AT, …) for amplitude is invalid: $exc")
            throw_error ? rethrow() : return false
        end
        if !(val isa Number)
            push!(errors, "evalcontrols(::$AT, …) for amplitude does not return a number")
        end
    else
        @error("evalcontrols(::$AT, …) for amplitude is not implemented")
        throw_error ? error("Invalid control amplitude") : return false
    end

    # getcontrolderiv
    #=
    if hasmethod(getcontrolderiv, (AT, eltype(controls)))
        for control in controls
            try
                d = getcontrolderiv(ampl, control)
                val = evalcontrols(d, vals_dict, tlist, 1)
                if !(val isa Number)
                    push!(
                        errors,
                        "getcontrolderiv(::$AT, …) for amplitude does not return something that evaluates to a number"
                    )
                end
            catch exc
                push!(errors, "getcontrolderiv(::$AT, …) for amplitude is invalid: $exc")
            end
        end
        v = rand()
        dummy_control = t -> v
        try
            d = getcontrolderiv(ampl, dummy_control)
            if d ≠ 0.0
                push!(
                    errors,
                    "getcontrolderiv(::$AT, …) for amplitude does not return `0.0` for a control that the amplitude does not depend on"
                )
            end
        catch exc
            push!(
                errors,
                "getcontrolderiv(::$AT, …)) for amplitude is invalid for a control that the generator does not depend on: $exc"
            )
        end
    else
        push!(errors, "getcontrolderiv(::$AT, …) for amplitude is not implemented")
    end
    =#

    if length(errors) > 0
        for error in errors
            @error(error)
        end
        throw_error ? error("Invalid control amplitude") : return false
    end

    return true

end


end
