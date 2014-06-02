#include <iostream>
#include <algorithm>

#include <ocl_wrapper.h>
#include <utl_utils.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#undef SIMPLE
#define LOCAL
#undef OUTPUT
#define EXECUTE_N_TIMES 1

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
__kernel void multiply_simple(unsigned int n, unsigned int k, unsigned int m,
                       __global TYPE *dst, __global TYPE *src1, __global TYPE *src2)
{
    unsigned int id0 = get_global_id(0);
    unsigned int id1 = get_global_id(1);

    if(id0 >= n || id1 >= m) return;

    unsigned int index1 = id0;
    unsigned int index2 = id1*k;
    unsigned int end = index2 + k;

    unsigned int dstindex = id0+id1*k;
    TYPE result = 0;

    while(index2 < end)
    {
        result += src1[index1] * src2[index2];
        index1+=n;
        index2++;
    }

    dst[dstindex] = result;
}

template<class TYPE>
__kernel void multiply(unsigned int n, unsigned int k, unsigned int m,
                       __global TYPE *dst, __global TYPE *src1, __global TYPE *src2)
{
    int BLOCK_SIZE = 16;
    __local TYPE As[16 * 16];
    __local TYPE Bs[16 * 16];

    unsigned int g_col = get_group_id(0);
    unsigned int g_row = get_group_id(1);

    unsigned int l_row = get_local_id(0);
    unsigned int l_col = get_local_id(1);

    if(g_col >= n / 16 || g_row >= m / 16 || l_col >= 16 || l_row >= 16)
        return;

    TYPE c_value = 0;

    for (int j = 0; j < (k / 16); ++j) {
        //
        As[l_col * 16 + l_row] = src1[(g_row * 16 + j * n * 16) + (l_col * n + l_row)];

        /*dst[(g_row * 16 + g_col * n * 16) + (l_col * n + l_row)]
                = src1[(g_row * 16 + g_col * n * 16) + (l_col * n + l_row)];*/

        Bs[l_col * 16 + l_row] = src2[(j * 16 + g_col * k * 16) + (l_col * k + l_row)];

        /*dst[(g_row * 16 + g_col * n * 16) + (l_col * n + l_row)]
                        = src2[(g_row * 16 + g_col * k * 16) + (l_col * k + l_row)];*/

        //
        barrier(CLK_LOCAL_MEM_FENCE);
        //
        for (int e = 0; e < 16; ++e) {
            c_value += Bs[l_col * 16 + e] * As[e * 16 + l_row];
        //
	}
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    dst[(g_row * 16 + g_col * n * 16) + (l_col * n + l_row)] = c_value;
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

        typedef float Type;
//         typedef utl::Matrix <Type,utl::column_major_tag> Matrix;
//         typedef utl::Ones <Type,utl::column_major_tag> Ones;
        typedef utl::Zeros <Type,utl::column_major_tag> Zeros;
        typedef utl::Rand <Type,utl::column_major_tag, utl::uniform_dist_tag> Rand;

        using Timer = utl::Timer<utl::MilliSeconds>;

        // get the kernels.
#ifdef LOCAL
        ocl::Kernel &kernel = program.kernel("multiply", utl::type::Single);
#endif
#ifdef SIMPLE
        ocl::Kernel &kernel_simple = program.kernel("multiply_simple", utl::type::Single);
#endif

#define SIZE 4096*2
        size_t n = SIZE, k = SIZE, m = SIZE;

        size_t elements_a = n * k;
        size_t size_bytes_a = elements_a * sizeof(Type);
	
	size_t elements_b = k * m;
        size_t size_bytes_b = elements_b * sizeof(Type);
	
	size_t elements_out = n * m;
        size_t size_bytes_out = elements_out * sizeof(Type);

        // set the index space for the kernels
#ifdef LOCAL
        kernel.setWorkSize(16, 16, n, m);
#endif
#ifdef SIMPLE
        kernel_simple.setWorkSize(16, 16, n, m);
#endif

        // create host matrices
        auto h_matrix_in_a  = Rand(n,k,1,5);
	auto h_matrix_in_b  = Rand(k,m,1,5);

        auto h_matrix_out = Zeros(n,m);
#ifdef SIMPLE
        auto h_matrix_out_simple = Zeros(n,m);
#endif

         //std::cout << "Matrix(a) before computation: " << std::endl << h_matrix_in_a << std::endl;
     	 //std::cout << "Matrix(b) before computation: " << std::endl << h_matrix_in_b << std::endl;
         //std::cout << "Matrix(out) before computation: " << std::endl << h_matrix_out << std::endl;

        // create device buffers on the specified context
        ocl::Buffer d_matrix_in_a (context, size_bytes_a);
	ocl::Buffer d_matrix_in_b (context, size_bytes_b);
        ocl::Buffer d_matrix_out(context, size_bytes_out);
        ocl::Buffer d_matrix_out_simple(context, size_bytes_out);

        // copy data from host buffers to device buffers
        d_matrix_in_a.write(queue, 0, h_matrix_in_a.data(), size_bytes_a);
	d_matrix_in_b.write(queue, 0, h_matrix_in_b.data(), size_bytes_b);
#ifdef LOCAL
    d_matrix_out.write(queue, 0, h_matrix_out.data(), size_bytes_out);
#endif
#ifdef SIMPLE
    d_matrix_out_simple.write(queue, 0, h_matrix_out_simple.data(), size_bytes_out);
#endif

        // EXECUTE_N_TIMES both kernels only if the event_write is completed.
        // note that kernel executions are always asynchronous.

#ifdef SIMPLE
    Timer::tic();
      kernel_simple(queue,
             int(n), int(k), int(m),
             d_matrix_out_simple.id(), d_matrix_in_a.id(), d_matrix_in_b.id());
      queue.finish();
    Timer::toc();

    d_matrix_out_simple.read(queue, h_matrix_out_simple.data(), size_bytes_out);

    // timer toc
    float min_gpu = std::min_element(h_matrix_out_simple.begin(), h_matrix_out_simple.end())[0];

    std::cout << "[INFO] Simple Minimum [GPU]: " << min_gpu << "\n[INFO] Simple time [GPU in us] = " << Timer::elapsed().count() << std::endl;
#endif
#ifdef LOCAL
	Timer::tic();
	for(size_t i = 0; i < EXECUTE_N_TIMES; ++i){
      kernel(queue,
             int(n), int(k), int(m),
             d_matrix_out.id(), d_matrix_in_a.id(), d_matrix_in_b.id());
	  queue.finish();
	}
	Timer::toc();
	
	// copy data from device buffers to host buffers
        d_matrix_out.read(queue, h_matrix_out.data(), size_bytes_out);


	
	// timer toc
#ifndef SIMPLE
    float
#endif
	min_gpu = std::min_element(h_matrix_out.begin(), h_matrix_out.end())[0];
	
	std::cout << "[INFO] Minimum [GPU]: " << min_gpu << "\n"
	<< "[INFO] Time [GPU in us] = " << Timer::elapsed().count()/EXECUTE_N_TIMES << " (executed "<<EXECUTE_N_TIMES << " time(s))" << std::endl;
#endif

	//auto h_matrix_correct = (h_matrix_in_a * h_matrix_in_b);
	
#ifdef SIMPLE
	#ifdef LOCAL
	    if(h_matrix_out == h_matrix_out_simple) {
	    std::cout << "[INFO] Computation was correct." << std::endl;
	    } else {
		std::cout << "[ERR ] FAILURE: Computation was incorrect!" << std::endl;
	    }

		#ifdef OUTPUT
		     std::cout << "Matrix(simple) after computation : " << std::endl << h_matrix_out_simple << std::endl;
		     std::cout << "Matrix(local)  after computation : " << std::endl << h_matrix_out << std::endl;
		#endif
	#else
		auto h_matrix_correct = (h_matrix_in_a * h_matrix_in_b);
		if(h_matrix_correct == h_matrix_out_simple) {
		       std::cout << "[INFO] Simple computation was correct." << std::endl;
	       } else {
		       std::cout << "[ERR ] FAILURE: Simple computation was incorrect!" << std::endl;
	       }
	#endif
#endif
    }


	return 0;
}
