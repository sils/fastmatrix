/******************************************************************************
 * Provides the implementation of a matrix class that makes use of GPUs or
 * multiple cores.
 * 
 * Author: Lasse Schuirmann
 ******************************************************************************/

#ifndef matrix_h
#error "This file is only to be included by the correct header file!"
#else

#include <matrix_kernels.h>
#include <debug.h>


template <typename TYPE>
Matrix<TYPE>::Matrix()
{
    PrepareGPU();
}

template<typename TYPE>
Matrix<TYPE>::Matrix(const unsigned int rows, const unsigned int cols, const TYPE initval)
{
    PrepareGPU();
}

template<typename TYPE>
void Matrix<TYPE>::PrepareGPU()
{
    m_platform = ocl::Platform(ocl::device_type::CPU);
#if VERB_TYPE_ACTIVE(PLATFORM_INFO)
    DEBUG_OUTPUT(PLATFORM_INFO_STR << "Info about the chosen platform:");
    m_platform.print();
#endif
    
    m_device = m_platform.device(ocl::device_type::CPU);
#if VERB_TYPE_ACTIVE(DEVICE_INFO)
    DEBUG_OUTPUT(DEVICE_INFO_STR << "Info about the chosen device:");
    m_device.print();
#endif
    
    //Prepare context
    m_context = ocl::Context(m_device);
    m_platform.insert(m_context);
    m_platform.setActiveContext(m_context);
    
    DEBUG_OUTPUT("Context is prepared.");

    //prepare queue
    m_queue.setContext(m_context);
    m_queue.setDevice(m_device);
    m_context.insert(&m_queue);
    m_context.setActiveQueue(m_queue);

    //Prepare program, build kernels
    m_program = ocl::Program(m_context, utl::Type::type<TYPE>());
    m_program << kernel_strings::kernels;
    m_program.build();
    if (m_program.isBuilt())
    {
        cout << "Program build successfully!" << endl;
    }

    m_program = ocl::Program(m_context, utl::Type::type<TYPE>());
    //m_context.setActiveProgram(program);
}

#if 0
 
    //Create program; get type objects for template parameters
    //TODO take the right types here
 
    
    ocl::Kernel &initKernel = program.kernel("init", utl::Type::type<TYPE>());
    
    ocl::Kernel &copyKernel = program.kernel("copy", utl::Type::type<TYPE>());

    ocl::Kernel &multiplyKernel = program.kernel("multiply", utl::Type::type<TYPE>());

    //TODO make the OCL objects private class members
#endif

#endif
