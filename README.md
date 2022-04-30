# Cudheart

[![time invested:](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb.svg)](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb)

## About the project
This project is a numpy-esque CUDA library to play around with vectors and matrices. 
<br>
The aim of this project is to provide a back end framework for a Tensorflow-esque library to be developed later.
<br>
Currently, no actual CUDA kernels are implemented, but they will be in the future.


- [ ] even more types for functions
- [ ] optimize the functions
- [ ] add more scenerios to shape mismatch exception
- [ ] test every function
- [ ] inplace in every function
- [ ] fix issue with cuda and massive vectors?
- [ ] benchmark cuda vs cpp
- [ ] make sure there is no undefined behavior when passing negative k (offset) to a function
- [ ] add built in casting for math functions with flags
- [ ] maybe more function variations?
- [ ] add complex numbers (Ai + B)
- [ ] add inf
- [ ] add negInf
- [ ] add NaN
- [ ] make util actually good for integrating cudheart in projects

p.s.: future me, please do remember to do all of the computations on flat vectors, and convert them "outside" of the math ty xoxo

maybe make the cuda also a part of the creation process?