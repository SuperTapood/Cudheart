# Roadmap for upcoming builds

## the feature complete build

<details>
	<summary> show details! </summary>

add all of the functions i plan to implement (at least on the numpy clone front)

any ticked funcs are added, but incomplete

- [X] solve
### String ops
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

</details>

## the tested build

<details>
	<summary> show details! </summary>

build unit tests for everything to iron out any major bugs

the modules that still need testing:

- [ ] Arrays
- [ ] Exceptions (light testing)
- [ ] Indexing
- [ ] Logic
- [ ] BaseMath
- [ ] Bitwise
- [ ] Complex
- [ ] Exps
- [ ] Linalg
- [ ] Statistics
- [ ] Trigo
- [ ] Random
- [ ] StringTypes


</details>

## the tasks build

<details>
	<summary> show details! </summary>

add and improve the library using the ideas i dumped onto the following list:

- [ ] integrate std functions for optimization
- [ ] revamp namespaces and make sure each one is self contained (does not contain stuff from other namespaces)
- [ ] convert m_str to private
- [ ] add getsize to string type
- [ ] make sure assert match size and assert match shape (what are you even doing bro) works the way its expected to work lol (default axis -1 checks restrictly, 0 for length = width, 1 for length = height)
- [ ] move string ops to where its counterparts are after the template overhaul
- [ ] add magic methods and redo functions where they can be used
- [ ] convert cuda and cpp pointers to shared memory
- [ ] add axis to everything
- [ ] properly name variables
- [ ] find a way to package this as a lib file
- [ ] add more bitwise function variation
- [ ] maybe add matrix and vector ops ndarray compatibility?
- [ ] add inline to all namespace functions (or maybe sort them into classes)
- [ ] change namespaces into classes
- [ ] add value guards to functions
- [ ] more overrides in random
- [ ] maybe add appendable vectors
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
- [ ] add default parameters to all functions
- [ ] add factorial(int) (which is the same as tgamma(int + 1))
- [ ] implement the numpy restrictions
- [ ] add more creation functions to ndarray, vector and matrix
- [ ] add swap function to ndarray family
- [ ] test the complex type compatibility with the rest of the functions
- [ ] make most of the functions applicable to the classes themselves (i.e. a.add(b) instead of add(a,b))
- [ ] make matrix ops not dependent on vector ops
- [ ] add more overloads to histogram2d


</details>

## the cuda build

<details>
	<summary> show details! </summary>

add a cuda alternative function for every function (where applicable)

the modules that still need cuda-ing:

- [ ] Arrays
- [ ] Indexing
- [ ] Logic
- [ ] BaseMath
- [ ] Bitwise
- [ ] Complex
- [ ] Exps
- [ ] Linalg
- [ ] Statistics
- [ ] Trigo
- [ ] Random
- [ ] StringTypes


</details>

## the cuda tested build

<details>
	<summary> show details! </summary>

build unit tests for cuda to iron out any major bugs and prepare Cudheart for release

the modules that still need cuda testing:

- [ ] Arrays
- [ ] (cuda specific) Exceptions (light testing)
- [ ] Indexing
- [ ] Logic
- [ ] BaseMath
- [ ] Bitwise
- [ ] Complex
- [ ] Exps
- [ ] Linalg
- [ ] Statistics
- [ ] Trigo
- [ ] Random
- [ ] StringTypes


</details>

## notes
<details> 
    <summary> show details! </summary>

this is how type enforcements work
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
    C* c = new C();
    int a = 5;
    float b = 5.0;
    // not pointer
    does_something_special_with_pointer(a);
    // not pointer
    does_something_special_with_pointer(b);
    // pointer
    does_something_special_with_pointer(c);
    // pointer
    does_something_special_with_pointer("hey");
    return 0;
}
```

use the code above with (https://www.boost.org/doc/libs/1_74_0/libs/math/doc/html/math_toolkit/result_type.html) and (https://github.com/boostorg/math/blob/b2538faaf9802af8e856c603b9001e33db826676/include/boost/math/tools/promotion.hpp) to do type forcing and enable the use of different types in a function


use unified memory instead of regular cuda memory to save time. (see refrences)

it takes a lot longer to invoke a kernel than it does to actually execute it, so its better to make kernels bigger than to invoke multiple of them. this is proven by how little difference there is between cuda and cpp implementations during an increase in complexity

references:
- https://programs.wiki/wiki/unified-virtual-addressing-unified-memory-addressing-for-memory-management.html
- https://developer.download.nvidia.com/CUDA/training/cuda_webinars_GPUDirect_uva.pdf


general links:
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- https://docs.nvidia.com/cuda/thrust/index.html
- https://www.run.ai/guides/nvidia-cuda-basics-and-best-practices/cuda-programming
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- https://www.olcf.ornl.gov/wp-content/uploads/2013/02/Intro_to_CUDA_C-TS.pdf
- https://www3.nd.edu/~zxu2/acms60212-40212/CUDA_C_Programming_Guide.pdf
- https://www.cs.utexas.edu/~rossbach/cs380p/papers/cuda-programming.pdf


</details>