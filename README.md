# Cudheart

total time spent on this dante-approved adventure: <br><br>
[![time invested:](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb.svg)](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb)

## About the project
This project is a numpy-esque CUDA and C++ library to play around with vectors and matrices. 
<br>
This library takes great inspiration and attempts to mimic the structure of python's numpy library.
<br>
The aim of this project is to provide a back-end framework for a Tensorflow-esque library to be developed at a later date.


- [ ] check and add the missing comments things
- [ ] even more types for functions
- [ ] remove fromVectorArray
- [ ] create vector using initializer list
- [ ] assert instead of print in the tests
- [ ] manually handle all possible exceptions (because cpp exceptions suck)
- [ ] don't flatten matrices unless its needed (or ur lazy or cuda is involved)
- [ ] deprecate containers and get rid of CUtil
- [ ] a bunch of new assertions and their exceptions
- [ ] optimize the functions (mainly for loops)
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