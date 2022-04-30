# list of functions left to implement

any ticked funcs are added, but incomplete

## Creation funcs
### general creation functions
- [X] logspace
- [X] geomspace
### matrix creation
- [X] tril
- [X] triu
- [X] vander

## array manipulation
## linalg
- [ ] inner
- [ ] outer
- [ ] matmul
- [ ] tensordot
- [ ] einsum
- [ ] einsumpath
- [ ] matrixPower
- [ ] kron
- [ ] cholesky
- [ ] qr
- [ ] svd
- [ ] eig
- [ ] eigh
- [ ] eigvals
- [ ] eigvalsh
- [ ] norm
- [ ] cond
- [ ] det
- [ ] matrixRank
- [ ] slogDet
- [ ] trace
- [ ] solve
- [ ] tensorsolve
- [ ] lstsq
- [ ] inv
- [ ] pinv
- [ ] tensorInv

## logic functions
- [ ] all
- [ ] any
- [ ] isFinite
- [ ] isInf
- [ ] isNaN
- [ ] isNegInf
- [ ] isPosInf
- [ ] isComplex
- [ ] isReal
- [ ] logicalAnd
- [ ] logicalOr
- [ ] logicalNot
- [ ] logicalXor
- [ ] allClose
- [ ] isClose
- [ ] arrrayEqual
- [ ] arrayEquiv
- [ ] greater
- [ ] greaterEqual
- [ ] less
- [ ] lessEqual
- [ ] equal
- [ ] notEqual

## math
### normal trigonometric funcs
- [ ] sin
- [ ] cos
- [ ] tan
- [ ] arcsin
- [ ] arccos
- [ ] arctan
- [ ] hypot
- [ ] arctan2
- [ ] degrees
- [ ] radians
- [ ] sinc
### hyperbolic
- [ ] sinh
- [ ] cosh
- [ ] tanh
- [ ] arcsinh
- [ ] arccosh
- [ ] arctanh
### rounding
- [ ] around
- [ ] round_
- [ ] rint
- [ ] fix
- [ ] floor
- [ ] ceil
- [ ] trunc
### sums and prods
- [ ] prod
- [ ] sum
- [ ] nanProd
- [ ] nanSum
- [ ] cumProd
- [ ] cumSum
- [ ] nanCumProd
- [ ] nanCumSum
### exps
- [ ] exp
- [ ] expm1
- [ ] exp2
- [ ] log
- [ ] log10
- [ ] log2
- [ ] log1p
- [ ] logaddexp
- [ ] logaddexp2
### fp routines
- [ ] signBit
- [ ] copySign
### rational
- [ ] lcm
- [ ] gcd
### arithmatic
- [ ] add
- [ ] reciprocal
- [ ] positive
- [ ] negative
- [ ] multiply
- [ ] divide
- [ ] power
- [ ] subtract
- [ ] trueDivide
- [ ] floorDivide
- [ ] floatPower
- [ ] fMod
- [ ] mod
- [ ] modF
- [ ] remainder
- [ ] divMod
### complex math
- [ ] angle
- [ ] real
- [ ] imag
- [ ] conj
- [ ] conjugate
### extrema finding
- [ ] maximum
- [ ] fMax
- [ ] aMax
- [ ] nanMax
- [ ] minimum
- [ ] fMin
- [ ] aMin
- [ ] nanMin
### misc
- [ ] convolve
- [ ] clip
- [ ] sqrt
- [ ] cbrt
- [ ] square
- [ ] absolute
- [ ] fabs
- [ ] sign
- [ ] heaviside
- [ ] nanToNum
- [ ] toBinary

## rng
### generator
- [ ] bitGenerator
- [ ] integers
- [ ] random
- [ ] choice
- [ ] bytes
- [ ] shuffle
- [ ] permutation
- [ ] permutated
- [ ] beta
- [ ] binomial
- [ ] chisquare
- [ ] dirichlet
- [ ] exponential
- [ ] f
- [ ] gamma
- [ ] geometric
- [ ] gumbel
- [ ] hypergeometric
- [ ] laplace
- [ ] logistic
- [ ] lognormal
- [ ] logseries
- [ ] multinomial
- [ ] multivariateHypergeometric
- [ ] multivariateNormal
- [ ] negativeBinomal
- [ ] noncentralChisquare
- [ ] noncentralF
- [ ] normal
- [ ] pareto
- [ ] poisson
- [ ] power
- [ ] rayleigh
- [ ] standardCauchy
- [ ] standardExponantial
- [ ] standardGamma
- [ ] standardNormal
- [ ] standardT
- [ ] triangular
- [ ] uniform
- [ ] vnmises
- [ ] wald
- [ ] weibull
- [ ] zipf

## sorting
- [ ] sort
- [ ] lexsort
- [ ] argsort
- [ ] sortComplex
- [ ] partition
- [ ] argpartition
## searching
- [ ] argmax
- [ ] nanargmax
- [ ] argmin
- [ ] nanargmin
- [ ] argwhere
- [ ] nonzero
- [ ] flatnonzero
- [ ] where
- [ ] searchsorted
- [ ] extract
- [ ] countNonzero
## statistics
- [ ] ptp
- [ ] percentile
- [ ] nanpercentile
- [ ] quantile
- [ ] nanquantile
- [ ] median
- [ ] average
- [ ] mean
- [ ] std
- [ ] var
- [ ] nanmedian
- [ ] nanmean
- [ ] nanstd
- [ ] nanvar
- [ ] corrcoef
- [ ] correlate
- [ ] cov
- [ ] histogram
- [ ] histogram2d
- [ ] bincount
- [ ] histogramBinEdges
- [ ] digitize


### from files? - not yet
- [ ] fromFile
- [ ] fromFunction
- [ ] fromIter
- [ ] fromString
- [ ] fromTxt
### joining and splitting arrays
- [ ] concatenate
- [ ] stack
- [ ] columnStack
- [ ] rowStack
- [ ] split
- [ ] tile
- [ ] repeat
### modify
- [ ] delete
- [ ] insert
- [ ] append
- [ ] resize
- [ ] trimZero
- [ ] unique

## String ops
- [ ] add
- [ ] multiply
- [ ] capitalize
- [ ] center
- [ ] decode
- [ ] encode
- [ ] expandTabs
- [ ] join
- [ ] lower
- [ ] lJust
- [ ] lStrip
- [ ] partition
- [ ] replace
- [ ] rJust
- [ ] RSplit
- [ ] RStrip
- [ ] strip
- [ ] translate
- [ ] upper
- [ ] equal
- [ ] notEqual
- [ ] greaterEqual
- [ ] lessEqual
- [ ] greater
- [ ] less
- [ ] count
- [ ] endsWith
- [ ] find
- [ ] index
- [ ] isNumeric
- [ ] isLower
- [ ] isUpper
- [ ] startsWith



# functions which have been declared redundent and their replacement:
 
- matmul => multiply
- multi_dot, vdot, inner for vectors => why even