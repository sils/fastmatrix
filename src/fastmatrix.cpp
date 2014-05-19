#include <iostream>
#include <algorithm>

#include <ocl_wrapper.h>
#include <utl_utils.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif


namespace kernel_strings {

const std::string kernels =
R"(

template<class Type>
__kernel void copy(int rows, int cols, __global Type *dst, __global Type *src)
{
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);

    if(id0 >= rows || id1 >= cols) return;

    int index = id0 + id1*rows;

    dst[index] = src[index];
}

template<class Type>
__kernel void addc(int rows, int cols, __global Type *dst, __global Type *src, Type c)
{
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);

    if(id0 >= rows || id1 >= cols) return;

    int index = id0 + id1*rows;

    dst[index] = src[index] + c;
}

template<class TYPE>
__kernel void multiply(unsigned int n, unsigned int k, unsigned int m,
                       __global TYPE *dst, __global TYPE *src1, __global TYPE *src2)
{
    unsigned int id0 = get_global_id(0);
    unsigned int id1 = get_global_id(1);

    if(id0 >= n || id1 >= m) return;
    
    unsigned int index1 = id0*k;
    unsigned int end = index1 + k;
    unsigned int index2 = id1;
    
    unsigned int dstindex = id0*m+id1;
    TYPE result = 0;
    
    while(index1 < end)
    {
        result += src1[index1] * src2[index2];
        index1++;
        index2+=m;
    }
    
    dst[dstindex] = result;
}

)";

}

int main()
{
    ocl::Platform platform(ocl::device_type::GPU);
    ocl::Device device = platform.device(ocl::device_type::GPU);

    // creates a context for a decice or platform
    ocl::Context context(device);

    // insert contexts into the platform
    platform.insert(context);

    // create command queue.
    ocl::Queue queue(context, device);


    // create program on a context.
    ocl::Program program(context, utl::type::Single | utl::type::Int);

    // insert kernels into the program.
    program << kernel_strings::kernels;

    // kernels are created and program is built for the context.
    program.build();
    {

        typedef int Type;
//         typedef utl::Matrix <Type,utl::column_major_tag> Matrix;
//         typedef utl::Ones <Type,utl::column_major_tag> Ones;
        typedef utl::Zeros <Type,utl::row_major_tag> Zeros;
        typedef utl::Rand <Type,utl::row_major_tag, utl::uniform_dist_tag> Rand;

        // get the kernels.
        ocl::Kernel &kernel = program.kernel("multiply", utl::type::Int);

        size_t n = 1024, k = 1024, m = 1024;

        size_t elements_a = n * k;
        size_t size_bytes_a = elements_a * sizeof(Type);
	
	size_t elements_b = k * m;
        size_t size_bytes_b = elements_b * sizeof(Type);
	
	size_t elements_out = n * m;
        size_t size_bytes_out = elements_out * sizeof(Type);

        // set the index space for the kernels
        kernel.setWorkSize(16, 16, n, m);

        // create host matrices
        auto h_matrix_in_a  = Rand(n,k,1,5);
	auto h_matrix_in_b  = Rand(k,m,1,5);
	
        auto h_matrix_out = Zeros(n,m);

        // std::cout << "Matrix(a) before computation: " << std::endl << h_matrix_in_a << std::endl;
	// std::cout << "Matrix(b) before computation: " << std::endl << h_matrix_in_b << std::endl;
        // std::cout << "Matrix(out) before computation: " << std::endl << h_matrix_out << std::endl;

        // create device buffers on the specified context
        ocl::Buffer d_matrix_in_a (context, size_bytes_a);
	ocl::Buffer d_matrix_in_b (context, size_bytes_b);
        ocl::Buffer d_matrix_out(context, size_bytes_out);

        // copy data from host buffers to device buffers
        d_matrix_in_a.write(queue, 0, h_matrix_in_a.data(), size_bytes_a);
	d_matrix_in_b.write(queue, 0, h_matrix_in_b.data(), size_bytes_b);

        // execute both kernels only if the event_write is completed.
        // note that kernel executions are always asynchronous.
	const size_t execute = 2;
	utl::Timer::tic();
	for(size_t i = 0; i < execute; ++i){
	  kernel(queue, int(n), int(k), int(m), d_matrix_out.id(), d_matrix_in_a.id(), d_matrix_in_b.id());
	  queue.finish();
	}
	utl::Timer::toc();
	
	// copy data from device buffers to host buffers
        d_matrix_out.read(queue, h_matrix_out.data(), size_bytes_out);
	
	// timer toc
	float min_gpu = std::min_element(h_matrix_out.begin(), h_matrix_out.end())[0];
	
	std::cout << "Minimum[GPU]: " << min_gpu << ", Time[GPU] = " << utl::Seconds(utl::Timer::elapsed(execute)) << std::endl;
	
	auto h_matrix_correct = (h_matrix_in_a * h_matrix_in_b);
	
        // std::cout << "Matrix(out) after computation : " << std::endl << h_matrix_out << std::endl;
	// std::cout << "Matrix after computation : " << std::endl << h_matrix_correct << std::endl;

        if(h_matrix_correct == h_matrix_out) std::cout << "Computation was correct." << std::endl;
    }


	return 0;
}
