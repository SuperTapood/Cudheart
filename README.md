# Cudheart

[![time invested:](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb.svg)](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb)

## About the project
This project is a numpy-esque CUDA library to play around with vectors and matrices. 
<br>
The aim of this project is to provide a back end framework for a Tensorflow-esque library to be developed later.
<br>
Currently, no actual CUDA kernels are implemented, but they will be in the future.


- [ ] fix the ops functions T nonsense
- [ ] add support for a negative k offset (ie fix undefined behavior when passing negative k to a function)
- [ ] add the rest of matrix creation functions
- [ ] test matrix creation and conversion
- [ ] add documentation
- [ ] add custom complex numbers (Ai + B)
- [ ] add inf
- [ ] add negInf
- [ ] add NaN
- [ ] add math virtual class
- [ ] add calc funcs
- [ ] add trigo funcs
- [ ] add bitwise funcs
- [ ] add comparison funcs
- [ ] even more documentation
- [ ] reverse the fucking matrix creation thing maybe

p.s.: future me, please do remember to do all of the computations on flat vectors, and convert them "outside" of the math ty xoxo

maybe make the cuda also a part of the creation process?