# Cudheart

[![time invested:](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb.svg)](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb)

## About the project
Cudheart is a numpy-esque library that aims to provide a simple and intuitive solution for CUDA based computing.
Currently only featuring C++ based functions, but CUDA stuff is coming I swear.


- [X] better fetching in ~~vector object~~ EVERYTHING
- [ ] **DOCUMENTATION**
- [X] arrays will now always copy given arrays
- [X] test direct vector creation with int and double
- [X] arrays now have multi dim indexing
- [X] hide array fields
- [ ] add the rest of the datatypes
- [ ] actual info for the module (version and stuff)
- [X] add shape object to store multi dim size
- [X] change from vector to array so i don't have a lot to change later
- [ ] add test and stuff for DChar and stuff
- [ ] add the rest of creation functions
- [ ] dtypes will now store a casted copy of the array to save time


## Help with issues
### come on man we both know you'll forget these things

#### value is 0 for some reason when casting from void*!
bruh check the casting. when you are casting from a raw (void) type
to an actual usable data type, it reads the data itself very weirdly 
(because it doesn't know what type it was casted before being void pointer)