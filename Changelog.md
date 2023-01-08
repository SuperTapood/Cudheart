## changelog:

- finally got rid of CUtil
- fixed fromString not actually parsing all the string data
- kinda fixed stringtype and complextype arrays not printing correctly
- fixed save writing object pointers rather than actual data
- fixed save not being compatible with the loading functions
- added missing load overload with only the file name
- added missing arrops include
- exceptions will now be automatically raised when created
- added a new argument to exceptions which allows you to create them without raising them automatically
- commented out the load functions because they were redundent
- removed shape initializing in vector initializer list constructor
- ndarray will now type check in its constructor
- vectors and matrices should now cast correctly ;)
- fixed concatenate lol
- fixed split function and improved its performance maybe
- fixed reshape :D
- removed axis argument from remove vector
- added axis default to 'remove' function
- fixed remove function
- improved (and maybe fixed?) trimzeros
- fixed the unique function
- tests now use namespaces instead of regions
- fixed vector casting
- removed the pointer only vector constructor
- fixed complex printing
- fixed casting for like the 20th time :\
- removed getabs and setabs from vector as they are no longer needed
- renamed shapeLike to reshape :)
- added overload to vector transpose
- replaced rotate with rot90 and fixed it

## notes:
- 