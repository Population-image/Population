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
#include <algorithm>

const int MATRIX_DIM = 1024;
const int BLOCK_SIZE = 32;
const int MAX_NB_THREADS = 1024; // GPU dependant
const int N_ROUNDS = 10;

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


struct ConvertRV32ToGrey{
    static bool init;
    static pop::UI8 _look_up_table[256][256][256];
    static pop::UI8 lumi(const pop::VecN<4,pop::UI8> &rgb){
        if(init==false){
            init= true;
            for(unsigned int i=0;i<256;i++){
                for(unsigned int j=0;j<256;j++){
                    for(unsigned int k=0;k<256;k++){
                        _look_up_table[i][j][k]=pop::ArithmeticsSaturation<pop::UI8,pop::F64>::Range(0.299*i + 0.587*j + 0.114*k+0.000001);
                    }
                }
            }
        }
        return _look_up_table[rgb(2)][rgb(1)][rgb(0)];
    }
};
bool ConvertRV32ToGrey::init =false;
pop::UI8 ConvertRV32ToGrey::_look_up_table[256][256][256];

template<typename T>
__device__ T arithmeticSaturationRangeGPU(pop::F64 p, T limitMin, T limitMax) {
    if(p>=limitMax) return limitMax;
    else if(p<limitMin) return limitMin;
    else return static_cast<T>(p);
}

template<typename T>
__global__ void lumi(T *initImage, T *greyImage, const int sizeGreyImage, const T limitMin, const T limitMax) {
	// thread tx of block bx: works on values from blockSize*bx + tx*4 to blockSize*bx + tx*4 + 3
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int start = idx<<2;

	if (idx < sizeGreyImage) {
		pop::F64 p = 0.000001;
		p += 0.299*initImage[start+2];
		p += 0.587*initImage[start+1];
		p += 0.114*initImage[start];
		greyImage[idx] = arithmeticSaturationRangeGPU(p, limitMin, limitMax);
	}
}

template<typename T>
void lumi_test() {
    uint64_t start_time, stop_time, diff_time;
    cudaEvent_t cuda_start_time, cuda_stop_time;
    cudaError_t error;
    float msecTotal;

    std::cout << "---\tWorking with " << type_name<T>() << "\t---" << std::endl;

    pop::MatN<2,pop::VecN<4,T> > h_image_RV32(MATRIX_DIM,MATRIX_DIM); // the initial image
    for (int i=0; i<MATRIX_DIM; i++) {
        for (int j=0; j<MATRIX_DIM; j++) {
        	pop::VecN<4,T> v;
        	v(0) = i%256;
        	v(1) = (255-(j%256));
        	v(2) = (i+j*10)%256;
        	v(3) = 0;
        	h_image_RV32(i, j) = v;
        }
    }

	pop::MatN<2, T> imggrey_pop(MATRIX_DIM, MATRIX_DIM); // the image after computation on the CPU
	pop::MatN<2, T> h_imggrey(MATRIX_DIM, MATRIX_DIM); // the image on host side after computation on the GPU


    /********************** CPU lumi **********************/
	rdtsc(start_time);
	for (int i=0; i<N_ROUNDS; i++) {
		std::transform(h_image_RV32.begin(),h_image_RV32.end(),imggrey_pop.begin(),ConvertRV32ToGrey::lumi);
	}
	rdtsc(stop_time);

	diff_time = diffTime(stop_time, start_time)/(N_ROUNDS*1000.0);
	std::cout << "CPU lumi: " << diff_time << "ms" << std::endl;


    /********************** CUDA initializations **********************/

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

    T* d_image_RV32;
    const int sizeInitialImage = h_image_RV32.size() * 4 * sizeof(T);
    cudaMalloc((void**)&d_image_RV32, sizeInitialImage);

    T* d_imggrey;
    const int sizeGreyImage = h_imggrey.size() * sizeof(T);
    cudaMalloc((void**)&d_imggrey, sizeGreyImage);

    /********************** GPU lumi **********************/

    error = cudaEventRecord(cuda_start_time, NULL);
    if (error != cudaSuccess)
    {
    	std::cerr << "Failed to record start event (error code " << cudaGetErrorString(error) << ")!" << std::endl;
    }

    for (int i=0; i<N_ROUNDS; i++) {
    	cudaMemcpy(d_image_RV32, h_image_RV32.data(), sizeInitialImage, cudaMemcpyHostToDevice);

    	lumi<<<h_imggrey.size()/MAX_NB_THREADS, MAX_NB_THREADS>>>(d_image_RV32, d_imggrey, h_imggrey.size(), pop::NumericLimits<T>::minimumRange(), pop::NumericLimits<T>::maximumRange());

    	error = cudaPeekAtLastError();
    	if (error != cudaSuccess) {
    		std::cerr << "There was an error executing the kernel: " << cudaGetErrorString(error) << std::endl;
    	}

        cudaMemcpy(h_imggrey.data(), d_imggrey, sizeGreyImage, cudaMemcpyDeviceToHost);
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
    std::cout << "GPU lumi: " << msecTotal / N_ROUNDS << "ms" << std::endl;


    /********************** Ok, done **********************/

    cudaMemcpy(h_imggrey.data(), d_imggrey, sizeGreyImage, cudaMemcpyDeviceToHost);
    cudaFree(d_image_RV32);
    cudaFree(d_imggrey);

    std::cout << "Is the result correct? " << (h_imggrey == imggrey_pop ? "true" : "false") << std::endl << std::endl;
    /*
    for (int i=0; i<MATRIX_DIM; i++) {
    	for (int j=0; j<MATRIX_DIM; j++) {
    		if (h_cudaResult(i, j) != h_imggrey(i, j)) {
    			std::cout << "Diff: c(" << i << ", " << j << ") = " << h_cudaResult(i, j) << "\t!=\tc_pop(" << i << ", " << j << ") = " << h_imggrey(i, j) << std::endl;
    		}
    	}
    }
    */
}

void cuda_test()
{
	init_clock_mhz();

	// Testing matrix multiply
	if (false) {
		multiply_matrix_test<pop::UI32>();
		multiply_matrix_test<pop::F32>();
		multiply_matrix_test<pop::F64>();
	}

	// cost of lumi
	lumi_test<pop::UI8>();
}
#endif
