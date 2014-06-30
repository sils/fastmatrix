
// c Zeros( M );
// A Ones ( M, N );
// b Ones ( N );


// thread m is responsible to calculate one inner product of row a and b.
// thread m, thread m+1 address contiguous column elements of A -> coalesced memory access
template<class Type>
__kernel void matvec1_cmajor(__global Type *c, __global Type *A, __global Type *b)
{
	int m = get_global_id(0); 

	if(m >= M) return;
	
	__global Type* a = A + m;

	Type s = 0;
	for(size_t n = 0; n < N; ++n)
		s += a[n*M] * b[n];
		
	c[m] = s;
}

// thread m is responsible to calculate one inner product of row a and b.
// thread m, thread m+1 address not contiguous column elements of A
template<class Type>
__kernel void matvec1_rmajor(__global Type *c, __global Type *A, __global Type *b)
{
	int m = get_global_id(0);

	if(m >= M) return;
	
	__global Type* a = A + m*N;

	Type s = 0;
	for(size_t n = 0; n < N; ++n)
		s += a[n] * b[n];
		
	c[m] = s;
}

// And here's our code!

template<class TYPE>
__kernel void multiplyc(__global TYPE *dst, __global TYPE *src1, __global TYPE *src2)
{
    __local TYPE As[W * W];
    __local TYPE Bs[W * W];

    unsigned int k = K;

    unsigned int g_col = get_group_id(0);
    unsigned int g_row = get_group_id(1);

    unsigned int l_row = get_local_id(0);
    unsigned int l_col = get_local_id(1);

    if(g_col >= N / W || g_row >= M / W || l_col >= W || l_row >= W)
        return;

    TYPE c_value = 0;

    for (int j = 0; j < (k / W); ++j) {
        //
        As[l_col * W + l_row] = src1[(g_row * W + j * N * W) + (l_col * N + l_row)];
        Bs[l_col * W + l_row] = src2[(j * W + g_col * k * W) + (l_col * k + l_row)];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int e = 0; e < W; ++e) {
            c_value += Bs[l_col * W + e] * As[e * W + l_row];
	}
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    dst[(g_row * W + g_col * N * W) + (l_col * N + l_row)] = c_value;
}

template<class TYPE>
__kernel void multiplyr(__global TYPE *dst, __global TYPE *src1, __global TYPE *src2)
{
    __local TYPE As[W][W];
    __local TYPE Bs[W][W];

    unsigned int g_col = get_group_id(0);
    unsigned int g_row = get_group_id(1);

    unsigned int l_col = get_local_id(0);
    unsigned int l_row = get_local_id(1);

    if(g_col >= N / W || g_row >= M / W || l_col >= W || l_row >= W)
        return;

    TYPE c_value = 0;
    
    for (int j = 0; j < (K / W); ++j) {
        As[l_row][l_col] = src1[(g_row * W + l_row) * K + (j * W + l_col)];
        Bs[l_row][l_col] = src2[(j * W + l_row) * N + (g_col * W + l_col)];

        // barrier(CLK_LOCAL_MEM_FENCE);
	
        for (int e = 0; e < W; ++e) {
            c_value += As[l_row][e] * Bs[e][l_col];
	}
        // barrier(CLK_LOCAL_MEM_FENCE);
    }
    // dst[0] = 1; // c_value;
    // dst[(g_row * W + l_row) * N + g_col * W + l_col] = 1; // c_value;
    dst[(g_row * W + l_row) * N + g_col * W + l_col] = c_value;
}

template<class TYPE>
__kernel void multiplycs(__global TYPE *dst, __global TYPE *src1, __global TYPE *src2)
{
    unsigned int id0 = get_global_id(0);
    unsigned int id1 = get_global_id(1);

    if(id0 >= N || id1 >= M) return;

    unsigned int index1 = id0;
    unsigned int index2 = id1*K;
    unsigned int end = index2 + K;

    unsigned int dstindex = id0+id1*K;
    TYPE result = 0;

    while(index2 < end)
    {
        result += src1[index1] * src2[index2];
        index1+=N;
        index2++;
    }

    dst[dstindex] = result;
}