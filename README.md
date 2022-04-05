# Cudheart

[![time invested:](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb.svg)](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb)

## About the project
This project is a numpy-esque CUDA library to play around with vectors and matrices. 
<br>
The goal of this project is to provide a back end framework for a Tensorflow-esque library to be developed later.
<br>
Currently, no actual CUDA kernels are implemented, but they will be in the future.


- [ ] add namespaces!
- [ ] add the rest of matrix creation functions
- [ ] test matrix creation and conversion
- [ ] add custom complex numbers (Ai + B)
- [ ] add documentation
- [ ] add inf
- [ ] add negInf
- [ ] add NaN
- [ ] add calc funcs
- [ ] add trigo funcs
- [ ] add bitwise funcs
- [ ] add comparison funcs


## Help with issues
### come on man we both know you'll forget these things

#### value is 0 when casting from void*!
bruh check the casting. when you are casting from a raw (void) type
to an actual usable data type, it reads the data itself very weirdly 
(because it doesn't know what type it was casted before being void pointer)

#### i can't cast from void* into TYPE*! it just says ```v was 0x[something]```!
when casting to void pointer you need to use a variable. if you directly cast
a value, it will threat its value as an address. please use:
```cpp
void func(void* value) {
    //something...
}

...

int main(){
    int value = 0;
    func(&value);
    // do not do this:
    // func((void*)0);
}
```
where: <br>
```0``` - the value you want to pass <br>
```func``` - the function that needs void pointer as argument