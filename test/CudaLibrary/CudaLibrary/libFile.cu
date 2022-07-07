#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "float.h"
#include <string>
#include <cmath>
#include "libHead.cuh"

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void my_kernel(unsigned arr1_sz, unsigned arr2_sz) {

    extern __shared__ char array[];

    double* my_ddata = (double*)array;
    char* my_cdata = arr1_sz * sizeof(double) + array;

    for (int i = 0; i < arr1_sz; i++) my_ddata[i] = (double)i * 1.1f;
    for (int i = 0; i < arr2_sz; i++) my_cdata[i] = (char)i;

    auto size = arr1_sz;
    if (size > arr2_sz) {
        size = arr2_sz;
    }

    for (int i = 0; i < size; i++) {
        printf("at offset %d, arr1: %lf, arr2: %d\n", i, my_ddata[i], (int)my_cdata[i]);
    }
}

__global__ void compute_it(float* data)
{
    int tid = threadIdx.x;
    __shared__ float myblock[1024];
    float tmp;

    // load the thread's data element into shared memory
    myblock[tid] = data[tid];

    // ensure that all threads have loaded their values into
    // shared memory; otherwise, one thread might be computing
    // on unitialized data.
    __syncthreads();

    // compute the average of this thread's left and right neighbors
    tmp = (myblock[tid > 0 ? tid - 1 : 1023] + myblock[tid < 1023 ? tid + 1 : 0]) * 0.5f;
    // square the previousr result and add my value, squared
    tmp = tmp * tmp + myblock[tid] * myblock[tid];

    // write the result back to global memory
    data[tid] = tmp;
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    unsigned double_array_size = 256;
    unsigned char_array_size = 128;
    unsigned shared_mem_size = (double_array_size * sizeof(double)) + (char_array_size * sizeof(char));
    my_kernel << <1, 1, shared_mem_size >> > (256, 128);
    cudaDeviceSynchronize();

    float* arr = new float[500];

    for (int i = 0; i < 500; i++) {
        arr[i] = 5;
    }

    compute_it << <1024, 1, 1 >> > (arr);

    for (int i = 0; i < 500; i++) {
        printf("%d: %f\n", i, arr[i]);
    }

    printf("hehee\n");
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    int f = add(1, 2);

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

 int DataType::toInt() {
    return (int)toLong();
}

 short DataType::toShort() {
    return (short)toLong();
}

 std::string DataType::toString() {
    return std::to_string(toDouble());
}

 DataInt8::DataInt8() {
    family = 1;
    member = 1;
    value = 0;
}
 short DataInt8::getFamily() {
    return family;
}
 short DataInt8::getMember() {
    return member;
}
 DataType* DataInt8::getVector(int size) {
    return new DataInt8[size];
}
 long DataInt8::toLong() {
    return value;
}
 double DataInt8::toDouble() {
    return (double)value;
}
 void DataInt8::setValue(long v) {
    value = v;
}

 Vector::Vector(int size, DataType* d) {
    m_size = size;
    m_data = d->getVector(size);
}

 Vector* Vector::add(Vector* other) {
    Vector* out = new Vector(m_size, &m_data[0]);

    for (int i = 0; i < m_size; i++) {
        out->m_data[i].setValue(m_data[0].toInt() + other->m_data[i].toInt());
    }

    return out;
}