/******************************************************************************
 * Provides the implementation of a matrix class that makes use of GPUs or
 * multiple cores.
 * 
 * Author: Lasse Schuirmann
 ******************************************************************************/

#ifndef matrix_h
#error "This file is only to be included by the correct header file!"
#else

#include <ocl_wrapper.h>
#include <utl_utils.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif


namespace kernel_strings
{

const std::string kernels =
R"(

template<class TYPE>
__kernel void copy(unsigned int rows, unsigned int cols,
                   __global TYPE *dst, __global TYPE *src)
{
    unsigned int id0 = get_global_id(0);
    unsigned int id1 = get_global_id(1);

    if(id0 >= cols || id1 >= rows) return;

    unsigned int index = id0 + id1*cols;

    dst[index] = src[index];
}

template<class TYPE>
__kernel void init(unsigned int rows, unsigned int cols,
                   __global TYPE *dst, __global TYPE initVal)
{
    unsigned int id0 = get_global_id(0);
    unsigned int id1 = get_global_id(1);

    if(id0 >= cols || id1 >= rows) return;

    unsigned int index = id0 + id1*cols;

    dst[index] = initVal;
}

template<class TYPE>
__kernel void multiply(unsigned int n, unsigned int k, unsigned int m,
                       __global TYPE *dst, __global TYPE *src1, __global TYPE *src2)
{
    unsigned int id0 = get_global_id(0);
    unsigned int id1 = get_global_id(1);

    if(id0 >= n || id1 >= m) return;
    
    unsigned int index1   = id0*k;
    unsigned int end      = index1 + k;
    unsigned int index2   = id1;
    unsigned int dstindex = id0*m+id1;
    TYPE     result   = 0;
    
    while(index1 < end)
    {
        result += src1[index1] * src2[index2];
        index1++;
        index2+=m;
    }
    
    dst[index1] = result;
}

)";

}

#endif

