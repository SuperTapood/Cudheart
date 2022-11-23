## changelog:

- finally got rid of CUtil
- fixed fromString not actually parsing all the string data
- kinda fixed stringtype and complextype arrays not printing correctly
- fixed save writing object pointers rather than actual data
- fixed save not being compatible with the loading functions
- added missing load overload with only the file name
- added unit tests for IO methods
- added missing arrops include
- exceptions will now be automatically raised when created
- added a new argument to exceptions which allows you to create them without raising them automatically
- commented out the load functions because they were redundent
- removed shape initializing in vector initializer list constructor
- ndarray will now type check in its constructor
- vectors and matrices should now cast correctly ;)

## notes:
- 