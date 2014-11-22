// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010

#include "popconfig.h"

#if defined(HAVE_CUDA)

#include "cuda_test.h"
#include "Population.h"
#include "microtime.h"

#include <iostream>
#include <string>
#include <unistd.h>
#include <cstdlib>
#include <cxxabi.h>

const int MATRIX_DIM = 1024;
const int BLOCK_SIZE = 32;


template<typename T>
std::string type_name()
{
    int status;
    std::string tname = typeid(T).name();
    char *demangled_name = abi::__cxa_demangle(tname.c_str(), NULL, NULL, &status);
    if(status == 0) {
        tname = demangled_name;
        std::free(demangled_name);
    }
    return tname;
}

template<typename TYPE>
__global__ void multiply_matrix_serial(TYPE *A, TYPE *B, TYPE *C) {
	for (int y=0; y<MATRIX_DIM; y++) {
		for (int x=0; x<MATRIX_DIM; x++) {
			TYPE sum = 0;
			for (int k=0; k<MATRIX_DIM; k++) {
				sum += A[x*MATRIX_DIM+k] * B[k*MATRIX_DIM+y];
			}
			C[x*MATRIX_DIM+y] = sum;
		}
	}
}

template<typename TYPE>
__global__ void multiply_matrix_parallel_global(TYPE *A, TYPE *B, TYPE *C) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int row_on_A = by * BLOCK_SIZE + ty;
	int start_on_A = MATRIX_DIM * row_on_A;
	__attribute__((unused)) int stop_on_A = start_on_A + MATRIX_DIM;

	int col_on_B = bx * BLOCK_SIZE + tx;
	int start_on_B = col_on_B;
	__attribute__((unused)) int stop_on_B = start_on_B + MATRIX_DIM * col_on_B;

	TYPE sum = 0;
	for (int k=0; k<MATRIX_DIM; k++) {
		sum += A[start_on_A+k] * B[start_on_B+k*MATRIX_DIM];
	}
	C[(by*BLOCK_SIZE+ty)*MATRIX_DIM+(bx*BLOCK_SIZE+tx)] = sum;
}


template<typename TYPE>
__global__ void multiply_matrix_parallel_shared(TYPE *A, TYPE *B, TYPE *C) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int cx = bx*BLOCK_SIZE+tx;
	int cy = (by*BLOCK_SIZE+ty)*MATRIX_DIM;

	TYPE sum = 0;

	for (int b=0; b<MATRIX_DIM; b+=BLOCK_SIZE) {
		__shared__ TYPE ABlock[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ TYPE BBlock[BLOCK_SIZE][BLOCK_SIZE];

		ABlock[ty][tx] = A[cy+(b+tx)];
		BBlock[ty][tx] = B[(b+ty)*MATRIX_DIM+cx];

		__syncthreads();

		//Ok, now we can do the sum
		for (int k=0; k<BLOCK_SIZE; k++) {
			sum += ABlock[ty][k] * BBlock[k][tx];
		}

		//needed to be sure that all the threads are ready for the next iteration
		__syncthreads();
	}

	C[cy+cx] = sum;
}

template<typename T>
void multiply_matrix_test() {
    uint64_t start_time, stop_time, diff_time;
    cudaEvent_t cuda_start_time, cuda_stop_time;
    cudaError_t error;
    float msecTotal;

    std::cout << "---\tWorking with " << type_name<T>() << "\t---" << std::endl;

    error = cudaEventCreate(&cuda_start_time);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to create start event (error code " << cudaGetErrorString(error) << ")!" << std::endl;
    }
    error = cudaEventCreate(&cuda_stop_time);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to create stop event (error code " << cudaGetErrorString(error) << ")!" << std::endl;
    }

    pop::MatN<2, T> h_A(MATRIX_DIM, MATRIX_DIM);
    for (int i=0; i<MATRIX_DIM; i++) {
        for (int j=0; j<MATRIX_DIM; j++) {
            h_A(i, j) = i+j*10;
        }
    }

    pop::MatN<2, T> h_B(MATRIX_DIM, MATRIX_DIM);
    for (int i=0; i<MATRIX_DIM; i++) {
        for (int j=0; j<MATRIX_DIM; j++) {
        	h_B(i, j) = (i==j ? 1 : 0);
        }
    }

    T *d_A, *d_B, *d_C;
    const int size = h_A.size()*sizeof(T);
    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_C, size);

    /********************** CPU multiplication **********************/

    pop::MatN<2, T> h_C_pop(MATRIX_DIM, MATRIX_DIM);

    rdtsc(start_time);
    h_C_pop = h_A * h_B;
    rdtsc(stop_time);

    diff_time = diffTime(stop_time, start_time)/1000.0;
    std::cout << "CPU matrix multiplication: " << diff_time << "ms" << std::endl;

    //Too slow to be activated
#if 0
    /********************** GPU multiplication, serial **********************/

    error = cudaEventRecord(cuda_start_time, NULL);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to record start event (error code " << cudaGetErrorString(error) << ")!" << std::endl;
    }

    multiply_matrix_serial<T><<<1, 1>>>(d_A, d_B, d_C);

    error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
    	std::cerr << "There was an error executing the kernel: " << cudaGetErrorString(error) << std::endl;
    }

    error = cudaEventRecord(cuda_stop_time, NULL);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to record stop event (error code " << cudaGetErrorString(error) << ")!" << std::endl;
    }

    error = cudaEventSynchronize(cuda_stop_time);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to synchronize on the stop event (error code " << cudaGetErrorString(error) << ")!" << std::endl;
    }

    msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, cuda_start_time, cuda_stop_time);
    std::cout << "GPU serial matrix multiplication: " << msecTotal << "ms" << std::endl;
#endif

#if 1
    /********************** GPU multiplication, parallel, global memory only **********************/

    dim3 grid(MATRIX_DIM/BLOCK_SIZE, MATRIX_DIM/BLOCK_SIZE, 1);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);

    error = cudaEventRecord(cuda_start_time, NULL);
    if (error != cudaSuccess)
    {
    	std::cerr << "Failed to record start event (error code " << cudaGetErrorString(error) << ")!" << std::endl;
    }

    multiply_matrix_parallel_global<T><<<grid, block>>>(d_A, d_B, d_C);

    error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
    	std::cerr << "There was an error executing the kernel: " << cudaGetErrorString(error) << std::endl;
    }

    error = cudaEventRecord(cuda_stop_time, NULL);
    if (error != cudaSuccess)
    {
    	std::cerr << "Failed to record stop event (error code " << cudaGetErrorString(error) << ")!" << std::endl;
    }

    error = cudaEventSynchronize(cuda_stop_time);
    if (error != cudaSuccess)
    {
    	std::cerr << "Failed to synchronize on the stop event (error code " << cudaGetErrorString(error) << ")!" << std::endl;
    }

    msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, cuda_start_time, cuda_stop_time);
    std::cout << "GPU parallel matrix multiplication (global): " << msecTotal << "ms" << std::endl;
#endif

#if 1
    /********************** GPU multiplication, parallel, shared memory **********************/

    error = cudaEventRecord(cuda_start_time, NULL);
    if (error != cudaSuccess)
    {
    	std::cerr << "Failed to record start event (error code " << cudaGetErrorString(error) << ")!" << std::endl;
    }

    multiply_matrix_parallel_shared<T><<<grid, block>>>(d_A, d_B, d_C);

    error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
    	std::cerr << "There was an error executing the kernel: " << cudaGetErrorString(error) << std::endl;
    }

    error = cudaEventRecord(cuda_stop_time, NULL);
    if (error != cudaSuccess)
    {
    	std::cerr << "Failed to record stop event (error code " << cudaGetErrorString(error) << ")!" << std::endl;
    }

    error = cudaEventSynchronize(cuda_stop_time);
    if (error != cudaSuccess)
    {
    	std::cerr << "Failed to synchronize on the stop event (error code " << cudaGetErrorString(error) << ")!" << std::endl;
    }

    msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, cuda_start_time, cuda_stop_time);
    std::cout << "GPU parallel matrix multiplication (shared): " << msecTotal << "ms" << std::endl;
#endif

    /********************** Ok, done **********************/

    pop::MatN<2, T> h_C(MATRIX_DIM, MATRIX_DIM);
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Is the result correct? " << (h_C == h_C_pop ? "true" : "false") << std::endl << std::endl;
    /*
    for (int i=0; i<MATRIX_DIM; i++) {
    	for (int j=0; j<MATRIX_DIM; j++) {
    		if (h_C(i, j) != h_C_pop(i, j)) {
    			std::cout << "Diff: c(" << i << ", " << j << ") = " << h_C(i, j) << "\t!=\tc_pop(" << i << ", " << j << ") = " << h_C_pop(i, j) << std::endl;
    		}
    	}
    }
    */
}

//What we should test:
//-scan
//-reduce
//-histogram

void cuda_test()
{
	init_clock_mhz();

	multiply_matrix_test<pop::UI32>();
	multiply_matrix_test<pop::F32>();
	multiply_matrix_test<pop::F64>();
}
#endif
