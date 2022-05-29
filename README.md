# Cudheart

total time spent on this dante-approved adventure: <br><br>
[![time invested:](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb.svg)](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb)

## About the project
This project is a numpy-esque CUDA C++ library to play around with vectors and matrices. 
<br>
This library takes great inspiration and attempts to mimic the structure of python's numpy library.
<br>
The aim of this project is to provide a back-end framework for a Tensorflow-esque library to be developed at a later date.

- [ ] add axis to everything
- [ ] add more bitwise function variation
- [ ] maybe add matrix and vector ops ndarray compatibility?
- [ ] add inline to all namespace functions (or maybe sort them into classes)
- [ ] change namespaces into classes
- [ ] add value guards to functions
- [ ] more overrides in random
- [ ] further generalize functions by passing shape
- [ ] make seed part of constants
- [ ] make random_bytes_engine part of constants
- [ ] move pi and euler into struct?
- [ ] fix a bunch of stupid matrix building and indexing bugs that are sure to be found
- [ ] fix the new object creation in complex math cpp
- [ ] make sure complex numbers won't be placed where they shouldn't (wherever std is involved)
- [ ] fix shape mismatch exception
- [ ] add map function to ndarray
- [ ] add overloads to simple random
- [ ] add shape argument to functions instead of using return types (+ initializer list overload?)
- [ ] add option to reduce exceptions into warnings and maybe even disable them (and choose whether they will be thrown or not)
- [ ] change exception color to red in the console
- [ ] add native math funcs to complex type
- [ ] add more overloads for expscpp
- [ ] add more overloads for basemathcpp
- [ ] remove all the useless overloads
- [ ] handle stupid exceptions (things like axis > 1 or axis < 0 things like that)
- [ ] fix docs in ndarray vector and matrix
- [ ] organise the function in matrix ndarray and vector
- [ ] get rid of util.cuh
- [ ] assert instead of print in the tests
- [ ] remove using
- [ ] even more type templates for functions
- [ ] remove fromVectorArray
- [ ] add cuda to logic funcs
- [ ] add cuda to linalg
- [ ] add cuda to trigo
- [ ] make sure every function is tested
- [ ] use max function to better print the matrix
- [ ] add readme to every module
- [ ] allow trigo funcs to accept angles with flags
- [ ] manually handle all possible exceptions (because cpp exceptions suck)
- [ ] fix vander
- [ ] add some meta data
- [ ] add shape equals to both matrix and vector
- [ ] replace cpp funcs in cuda namespaces with using
- [ ] divide functions into pragma regions
- [ ] fix non numeric vectors and matrices
- [ ] add reshape function
- [ ] match docs with numpy with the first matrix ops funcs
- [ ] fix index error for getting info from flat matrix to say matrix
- [ ] don't flatten matrices unless its needed (or ur lazy or cuda is involved)
- [ ] convert project to static library project
- [ ] use ndarray to reduce number of overrides
- [ ] fix internal casting in meshgrid
- [ ] add another type argument for set and get methods
- [ ] fix docs typos
- [ ] make matrix a vector of vectors to handle row fetching faster
- [ ] add axis for mixed operations (vectors and matrices) where usually a horizontal vector is assumed
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
- [ ] add tests from numpy site
- [ ] maybe add scalar overloads
- [ ] make sure ndarray objects are called with new using new overloads and shit
- [ ] add assert match size and insert it where needed during axis update
- [ ] add default parameters to functions

p.s.: future me, please do remember to do all of the computations on flat vectors, and convert them "outside" of the math ty xoxo


p.s.s: this is how type enforcements work
```cpp
#include <type_traits>
#include <iostream>
using namespace std;

//Overload for when T is not a pointer type
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value>::type
does_something_special_with_pointer (T t) {
    //Do something boring
    cout << "not pointer\n";
    // cout << t << endl;
}

//Overload for when T is a pointer type
template <typename T>
typename std::enable_if<!std::is_arithmetic<T>::value>::type 
does_something_special_with_pointer (T t) {
    //Do something special
    cout << "pointer\n";
    // cout << t << endl;
}


class C {
    
};

int main()
{
    cout<<"Hello World\n";
    C* c = new C();
    int a = 5;
    float b = 5.0;
    does_something_special_with_pointer(a);
    does_something_special_with_pointer(b);
    does_something_special_with_pointer(c);
    return 0;
}

```

use the code above with (https://www.boost.org/doc/libs/1_74_0/libs/math/doc/html/math_toolkit/result_type.html) and (https://github.com/boostorg/math/blob/b2538faaf9802af8e856c603b9001e33db826676/include/boost/math/tools/promotion.hpp) to do type forcing and enable the use of different types in a function