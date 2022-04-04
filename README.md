# Cudheart

[![time invested:](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb.svg)](https://wakatime.com/badge/user/8b4f0bdc-5133-4fba-98d4-d75498fa71f2/project/eccaf13a-dd3b-426e-b047-82a0bd7cc1eb)

## About the project
///Cudheart is a numpy-esque library that aims to provide a simple and intuitive solution for CUDA based computing.
<br>
/////Currently only featuring C++ based functions, but CUDA stuff is coming I swear.


- [ ] fix matrix print thing, cast vectors and matrices, add ndarray master interface for no reason other for maybe tensors down the line, maybe create wrappers so user doesn't have to play with datatypes, fix readme

/// incorrect please fix
- [ ] finish triu
- [ ] test the dll file in another seperate project and get that working
- [ ] test all creation functions
- [ ] actual info for the module (version and stuff)
- [ ] **DOCUMENTATION**
- [ ] add the rest of the datatypes
- [ ] add test and stuff for DChar and stuff
- [ ] dtypes will now store a casted copy of the array to save time when casting


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