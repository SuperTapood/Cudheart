#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "float.h"
#include <string>
#pragma comment(lib, "CudaLibrary")

__global__ void addKernel(int* c, const int* a, const int* b);

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);


class DataType {
public:
    virtual DataType* getVector(int size) = 0;
    virtual long toLong() = 0;
    int toInt();
    short toShort();
    virtual double toDouble() = 0;
    virtual std::string toString();
    virtual void setValue(long v) = 0;
    virtual short getFamily() = 0;
    virtual short getMember() = 0;
};

class DataInt8 : public DataType {
public:
    // -1 unsigned, 0 string type, 1 signed
    short family;
    // sorted based on size (0 indexed)
    short member;
    int value;
    DataInt8();
    short getFamily();
    short getMember();
    DataType* getVector(int size);
    long toLong();
    double toDouble();
    void setValue(long v);
};

class Vector {
public:
    int m_size;
    DataType* m_data;
    Vector(int size, DataType* d);
    Vector* add(Vector* other);
};

template <typename T>
T add(T a, T b) {
    return a + b;
}