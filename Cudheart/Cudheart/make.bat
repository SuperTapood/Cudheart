@echo off
nvcc -o exec Cudheart.cpp test.cpp
echo
echo running program:
exec
