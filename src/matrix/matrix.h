/**
 * Provides a matrix class that makes use of GPUs or multiple cores.
 * 
 * Author: Lasse Schuirmann
 */

#ifndef matrix_h
#define matrix_h

#include <iostream>

#include <ocl_wrapper.h>
#include <utl_utils.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

/**
 * \class Matrix
 * \brief Matrix class capable of using the GPU.
 */
template <typename TYPE>
class Matrix
{
public:
    /**
     * \brief Simplest constructor.
     *        Constructs an empty matrix.
     */
    Matrix();
    
    /**
     * \brief Copyconstructor.
     *        Makes a deep copy of the given argument.
     * 
     * \param copy      The matrix to be copied.
     */
    Matrix(const Matrix<TYPE> &copy);
    
    /**
     * \brief Constructor which reserves space in memory for the matrix.
     *        The values will not be initialized!
     * 
     * \param rows      Row count
     * \param cols      Column count
     */
    Matrix(const unsigned int rows, const unsigned int cols);
    
    /**
     * \brief Constructor which initializes the matrix.
     *        The values will be initialized according to the third parameter.
     * 
     * \param rows      Row count
     * \param cols      Column count
     * \param initval   Initial value for all cells
     */
    Matrix(const unsigned int rows, const unsigned int cols, const TYPE initval);
    
    //TODO overload operators (don't forget const)
private:
    /**
     * @brief Compiles kernels, gets devices and so on.
     */
    void PrepareGPU();

    ocl::Platform m_platform;
    ocl::Device   m_device;
    ocl::Context  m_context;
    ocl::Program  m_program;
};

//This is where the code lies!
#include "matrix_code.h"

#endif /* matrix_h */

