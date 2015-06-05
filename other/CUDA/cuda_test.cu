// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010

#include "popconfig.h"

#if defined(HAVE_CUDA)

#include <iostream>
#include <string>
#include <unistd.h>
#include <cstdlib>
#include <cxxabi.h>
#include <algorithm>
#include <cublas_v2.h>

#include "cuda_test.h"
#include "popcuda.h"
#include "Population.h"
#include "microtime.h"

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

void test_cublas(void) {
	cublasStatus_t	stat;
	cublasHandle_t handle;

	const int width = 3;
	const int height = 2;

	float* W = new float[width*height];
	W[0] = .5; W[1] = .8; W[2] = 2;
	W[3] = 1.5; W[4] = 1; W[5] = 0;

	float* dW = new float[width*height];
	dW[0] = 1; dW[1] = 2; dW[2] = 3;
	dW[3] = 4; dW[4] = 5; dW[5] = 6;

	const float eta = 0.9;

	std::cout << "***** BEFORE:" << std::endl;
	std::cout << "W = [" << std::endl;
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			std::cout << " " << W[i*width+j];
		}
		std::cout << std::endl;
	}
	std::cout << "]" << std::endl;

	std::cout << "dW = [" << std::endl;
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			std::cout << " " << dW[i*width+j];
		}
		std::cout << std::endl;
	}
	std::cout << "]" << std::endl;

	//*************************************************

	float* d_W;
	cudaMalloc(&d_W, width*height*sizeof(*d_W));
	cudaMemcpy(d_W, W, width*height*sizeof(*d_W), cudaMemcpyHostToDevice);

	float* d_dW;
	cudaMalloc(&d_dW, width*height*sizeof(*d_dW));
	cudaMemcpy(d_dW, dW, width*height*sizeof(*d_dW), cudaMemcpyHostToDevice);

	cublasCreate(&handle);

	float alpha = 1.0f;
	float beta = -eta;
	//  C = α op(A) + β op(B) -> W = 1 op(W) + -eta op(dW)
	stat = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, height, &alpha, d_W, width, &beta, d_dW, width, d_W, width);
	std::cout << "cublas status: " << popcuda::cublasGetErrorString(stat) << std::endl;

	cudaMemcpy(W, d_W, width*height*sizeof(*d_W), cudaMemcpyDeviceToHost);
	cudaMemcpy(dW, d_dW, width*height*sizeof(*d_dW), cudaMemcpyDeviceToHost);

	cublasDestroy(handle);
	cudaFree(d_dW);
	cudaFree(d_W);

	//*************************************************

	std::cout << "***** AFTER:" << std::endl;
	std::cout << "W = [" << std::endl;
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			std::cout << " " << W[i*width+j];
		}
		std::cout << std::endl;
	}
	std::cout << "]" << std::endl;

	std::cout << "dW = [" << std::endl;
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			std::cout << " " << dW[i*width+j];
		}
		std::cout << std::endl;
	}
	std::cout << "]" << std::endl;

	delete[] dW;
	delete[] W;

#if 0
	cublasStatus_t	stat;
	cublasHandle_t handle;

	const int width = 3;
	const int height = 2;
	float* W = new float[width*height];
	for (int i=0; i<width*height; i++) {
		W[i] = i;
	}

	std::cout << "W = [";
	for (int i=0; i<width*height; i++) {
		std::cout << " " << W[i];
	}
	std::cout << " ]" << std::endl;

	float* X = new float[width];
	X[0] = 2;
	for (int i=1; i<width; i++) {
		X[i] = 1;
	}

	std::cout << "X = [";
	for (int i=0; i<width; i++) {
		std::cout << " " << X[i];
	}
	std::cout << " ]" << std::endl;

	float* Y = new float[height];
	for (int i=1; i<height; i++) {
		Y[i] = 0;
	}

	float* d_W;
	cudaMalloc(&d_W, width*height*sizeof(*d_W));
	cudaMemcpy(d_W, W, width*height*sizeof(*d_W), cudaMemcpyHostToDevice);

	float* d_X;
	cudaMalloc(&d_X, width*sizeof(*d_X));
	cudaMemcpy(d_X, X, width*sizeof(*d_X), cudaMemcpyHostToDevice);

	float* d_Y;
	cudaMalloc(&d_Y, height*sizeof(*d_Y));

	cublasCreate(&handle);

	float alpha = 1.0f;
	float beta = 0.0f;
	//  Y = α op(W) X + β Y
	stat = cublasSgemv_v2(handle, CUBLAS_OP_T, width, height, &alpha, d_W, width, d_X, 1, &beta, d_Y, 1);	// Y = [ 3 15 ]
	std::cout << "cublas status: " << popcuda::cublasGetErrorString(stat) << std::endl;

	cudaMemcpy(Y, d_Y, height*sizeof(*d_Y), cudaMemcpyDeviceToHost);

	std::cout << "Y = [";
	for (int i=0; i<height; i++) {
		std::cout << " " << Y[i];
	}
	std::cout << " ]" << std::endl;

	cublasDestroy(handle);
	cudaFree(d_W);
	cudaFree(d_X);
	cudaFree(d_Y);

	delete[] Y;
	delete[] X;
	delete[] W;
#endif
}

template<typename T>
T convolution_cpu(const T in, const T kernel, const unsigned int r, const unsigned int step) {
	const unsigned d = 2*r;
	T out((in.sizeI()-d)/step, (in.sizeJ()-d)/step);

	std::cout << "Size of in: " << in.sizeI() << "x" << in.sizeJ() << std::endl;
	std::cout << "Size of kernel: " << kernel.sizeI() << "x" << kernel.sizeJ() << ", r = " << r << std::endl;
	std::cout << "Size of out: " << out.sizeI() << "x" << out.sizeJ() << ", S = " << step << std::endl;

	for (unsigned int i=0; i<out.sizeI(); i++) {
		for (unsigned int j=0; j<out.sizeJ(); j++) {
			float sum = 0.0;
			for (unsigned int n=0; n<=d; n++) {
				for (unsigned int m=0; m<=d; m++) {
					sum += in(n+i*step, m+j*step) * kernel(n, m);
				}
			}
			out(i, j) = sum;
		}
	}

	return out;
}

template<typename T>
__global__ void kernel_convolution_gpu(T* d_in, const unsigned int in_height, const unsigned int in_width, T* d_kernel, T* d_out, const unsigned int d, const unsigned int step, const unsigned int out_height, const unsigned int out_width) {
	unsigned int i = blockIdx.x*blockDim.y + threadIdx.x;
	unsigned int j = blockIdx.y*blockDim.x + threadIdx.y;

	if (i>=out_height || j>=out_width) {
		return;
	}

	float sum = 0.0;
	for (unsigned int n=0; n<=d; n++) {
		for (unsigned int m=0; m<=d; m++) {
			sum += d_in[(n+i*step)*in_width+(m+j*step)] * d_kernel[n*(d+1)+m];
		}
	}
	d_out[i*out_width+j] = sum;
}

#undef SHARED_IN

template<typename T>
__global__ void kernel_convolution_gpu_shared(T* d_in, const unsigned int in_height, const unsigned int in_width, T* d_kernel, T* d_out, const unsigned int d, const unsigned int step, const unsigned int out_height, const unsigned int out_width, unsigned int memsize) {
	unsigned int i = blockIdx.x*blockDim.y + threadIdx.x;
	unsigned int j = blockIdx.y*blockDim.x + threadIdx.y;

	if (i>=out_height || j>=out_width) {
		return;
	}

	extern __shared__ T h[];

	// the kernel is small, so it is faster enough to run this single-threaded copy
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for (unsigned int k=0; k<(d+1)*(d+1); k++) {
			h[k] = d_kernel[k];
		}
	}

#ifdef SHARED_IN
	T* shared_in = h + (d+1)*(d+1);
	unsigned int i_min = blockIdx.x*blockDim.y*step;
	unsigned int i_max = (blockIdx.x+1)*blockDim.y*step;
	unsigned int j_min = blockIdx.y*blockDim.x*step;
	unsigned int j_max = (blockIdx.y+1)*blockDim.x*step;

	//nb elements on x : i_max+d - i_min
	// per thread: (i_max+d - i_min)/blockDim.x
	// thread 0: from i_min to i_min+blockDim.x
	// thread 1: from i_min+blockDim.x to i_min+2*blockDim.x
	// ...
	// thread t: from i_min*step+t*blockDim.x to i_min*step+(t+1)*blockDim.x
	for (unsigned int k=i_min+threadIdx.x*blockDim.x; k<i_min+(threadIdx.x+1)*blockDim.x && k<i_max+d; k++) {
		for (unsigned int l=j_min+threadIdx.y*blockDim.y; l<j_max+(threadIdx.y+1)*blockDim.y && l<j_max+d; l++) {
			shared_in[(k-i_min)*(blockDim.x+d)+l-j_min] = d_in[k*in_width+l];
		}
	}
#endif

	__syncthreads();

	float sum = 0.0;
	for (unsigned int n=0; n<=d; n++) {
		for (unsigned int m=0; m<=d; m++) {
			//sum += d_in[(n+i*step)*in_width+(m+j*step)] * d_kernel[n*(d+1)+m];
			//sum += d_in[(n+i*step)*in_width+(m+j*step)] * h[n*(d+1)+m];
#ifdef SHARED_IN
			sum += shared_in[(n+i*step-i_min)*(blockDim.x+d)+(m+j*step-j_min)] * h[n*(d+1)+m];
#else
			sum += d_in[(n+i*step)*in_width+(m+j*step)] * h[n*(d+1)+m];
#endif
		}
	}
	d_out[i*out_width+j] = sum;
}

template<typename T>
pop::MatN<2, T> convolution_gpu(const pop::MatN<2, T> in, const pop::MatN<2, T> kernel, const unsigned int r, const unsigned int step) {
	cudaEvent_t cuda_start_time, cuda_stop_time;
	cudaError_t error;
	float msecTotal;

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

	const unsigned d = 2*r;
	pop::MatN<2, T> out((in.sizeI()-d)/step, (in.sizeJ()-d)/step);

	std::cout << "Size of in: " << in.sizeI() << "x" << in.sizeJ() << std::endl;
	std::cout << "Size of kernel: " << kernel.sizeI() << "x" << kernel.sizeJ() << ", r = " << r << std::endl;
	std::cout << "Size of out: " << out.sizeI() << "x" << out.sizeJ() << ", S = " << step << std::endl;

	T* d_in;
	cudaMalloc(&d_in, in.sizeI() * in.sizeJ() * sizeof(*d_in));
	cudaMemcpy(d_in, in.data(), in.sizeI() * in.sizeJ() * sizeof(*d_in), cudaMemcpyHostToDevice);

	T* d_kernel;
	cudaMalloc(&d_kernel, kernel.sizeI() * kernel.sizeJ() * sizeof(*d_kernel));
	cudaMemcpy(d_kernel, kernel.data(), kernel.sizeI() * kernel.sizeJ() * sizeof(*d_kernel), cudaMemcpyHostToDevice);

	T* d_out;
	cudaMalloc(&d_out, out.sizeI() * out.sizeJ() * sizeof(*d_out));

	const unsigned int max_nb_threads = popcuda::getMaxNumberThreadsPerBlock();

	dim3 block(out.sizeI(), out.sizeJ(), 1);
	dim3 grid(1, 1, 1);
	if (out.sizeI()*out.sizeJ() > max_nb_threads) {
		block.x = block.y = BLOCK_SIZE;
		grid.x = out.sizeI() / BLOCK_SIZE + (out.sizeI()%BLOCK_SIZE ? 1 : 0);
		grid.y = out.sizeJ() / BLOCK_SIZE + (out.sizeJ()%BLOCK_SIZE ? 1 : 0);
	}
	//std::cout << "grid = (" << grid.x << ", " << grid.y << "), block = (" << block.x << ", " << block.y << ")" << std::endl;

	error = cudaEventRecord(cuda_start_time, NULL);
	if (error != cudaSuccess)
	{
		std::cerr << "Failed to record start event (error code " << cudaGetErrorString(error) << ")!" << std::endl;
	}

	unsigned int memsize = kernel.sizeI()*kernel.sizeJ()*sizeof(*d_kernel);
#ifdef SHARED_IN
	memsize += (BLOCK_SIZE+d)*(BLOCK_SIZE+d)*sizeof(*d_out);
#endif
	if (memsize > popcuda::getMaxSharedMemPerBlock()) {
		std::cout << "Kernel too big for the shared memory version. Using the global memory version" << std::endl;
		kernel_convolution_gpu<<<grid, block>>>(d_in, in.sizeI(), in.sizeJ(), d_kernel, d_out, d, step, out.sizeI(), out.sizeJ());
	} else {
		kernel_convolution_gpu_shared<<<grid, block, memsize>>>(d_in, in.sizeI(), in.sizeJ(), d_kernel, d_out, d, step, out.sizeI(), out.sizeJ(), memsize);
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
	std::cout << "GPU convolution: " << msecTotal << "ms" << std::endl;

	cudaMemcpy(out.data(), d_out, out.sizeI() * out.sizeJ() * sizeof(*d_out), cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_kernel);
	cudaFree(d_in);

	return out;
}

void test_convolution(void) {
	uint64_t start_time, stop_time, diff_time;
	init_clock_mhz();

#if 0
	pop::Mat2F32 m(30, 30);
	for (int i=0; i<30; i++) {
		for (int j=0; j<30; j++) {
			m(i, j) = i*30+j;
		}
	}

	pop::Mat2F32 kernel_identity(3, 3);
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			kernel_identity(i, j) = (i==1 && j==1);
		}
	}

	const unsigned int nb_iter = 1;

	pop::Mat2F32 out_cpu;
	rdtsc(start_time);
	for (int i=0; i<nb_iter; i++)
		out_cpu = convolution_cpu(m, kernel_identity, 1, 1);
	rdtsc(stop_time);
	diff_time = diffTime(stop_time, start_time)/nb_iter;
	std::cout << "\t--> CPU convolution: " << diff_time << "usec <--\n" << std::endl;

	pop::Mat2F32 out_gpu;
	rdtsc(start_time);
	for (int i=0; i<nb_iter; i++)
		out_gpu = convolution_gpu(m, kernel_identity, 1, 1);
	rdtsc(stop_time);
	diff_time = diffTime(stop_time, start_time)/nb_iter;
	std::cout << "\t--> GPU convolution: " << diff_time << "usec <--\n" << std::endl;

	if (out_cpu != out_gpu) {
		std::cerr << "Erreur de calcul !" << std::endl;
	} else {
		std::cout << "Calcul correct !" << std::endl;
	}
#endif

#if 0
	pop::Mat2UI8 m8int;
	m8int.load("/home/pl/workspace/Population/image/Vd-Orig.png");
	pop::Mat2F32 m(m8int);

	pop::Mat2F32 kernel(3, 3);
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			kernel(i, j) = (i==1 && j==1 ? 8 : -1);
		}
	}

	rdtsc(start_time);
	pop::Mat2F32 out_cpu = convolution_cpu(m, kernel, 1, 1);
	rdtsc(stop_time);
	diff_time = diffTime(stop_time, start_time)/1000.0;
	std::cout << "\t--> CPU convolution: " << diff_time << "ms <--\n" << std::endl;

	rdtsc(start_time);
	pop::Mat2F32 out_gpu = convolution_gpu(m, kernel, 1, 1);
	rdtsc(stop_time);
	diff_time = diffTime(stop_time, start_time)/1000.0;
	std::cout << "\t--> GPU convolution: " << diff_time << "ms <--\n" << std::endl;

	if (out_cpu != out_gpu) {
		std::cerr << "Erreur de calcul !" << std::endl;
		/*
		for (int i=0; i<out_cpu.sizeI(); i++) {
			for (int j=0; j<out_cpu.sizeJ(); j++) {
				if (out_cpu(i, j) != out_gpu(i, j)) {
					std::cerr << "(" << i << ", " << j << "): " << static_cast<int>(out_cpu(i, j)) << " != " << static_cast<int>(out_gpu(i, j)) << std::endl;
				}
			}
		}
		*/
	} else {
		std::cout << "Calcul correct !" << std::endl;
	}

	pop::Mat2UI8 outui8(out_gpu);
	outui8.save("/home/pl/workspace/Population/image/Vd-convolution.jpg");
#endif

#if 1
	pop::Mat2UI8 m;
	m.load("/home/pl/workspace/Population/image/NYC.jpg");

	pop::Mat2UI8 kernel_identity(3, 3);
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			kernel_identity(i, j) = (i==1 && j==1);
		}
	}

	rdtsc(start_time);
	pop::Mat2UI8 out_cpu = convolution_cpu(m, kernel_identity, 1, 1);
	rdtsc(stop_time);
	diff_time = diffTime(stop_time, start_time)/1000.0;
	std::cout << "\t--> CPU convolution: " << diff_time << "ms <--\n" << std::endl;

	rdtsc(start_time);
	pop::Mat2UI8 out_gpu = convolution_gpu(m, kernel_identity, 1, 1);
	rdtsc(stop_time);
	diff_time = diffTime(stop_time, start_time)/1000.0;
	std::cout << "\t--> GPU convolution: " << diff_time << "ms <--\n" << std::endl;

	if (out_cpu != out_gpu) {
		std::cerr << "Erreur de calcul !" << std::endl;
		for (int i=0; i<out_cpu.sizeI(); i++) {
			for (int j=0; j<out_cpu.sizeJ(); j++) {
				if (out_cpu(i, j) != out_gpu(i, j)) {
					std::cerr << "(" << i << ", " << j << "): " << static_cast<int>(out_cpu(i, j)) << " != " << static_cast<int>(out_gpu(i, j)) << std::endl;
				}
			}
		}
	} else {
		std::cout << "Calcul correct !" << std::endl;
	}

	out_cpu.save("/home/pl/workspace/Population/image/NYC-cpu.jpg");
	out_gpu.save("/home/pl/workspace/Population/image/NYC-gpu.jpg");
#endif

#if 0
	bool cpu = false;

	std::cout << "Test convolution on " << (cpu ? "CPU" : "GPU") << std::endl;
	std::cout << std::endl;

	pop::Mat2F32 in(3, 3);
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			in(i, j) = i*3+j+1;
		}
	}

	pop::Mat2F32 kernel(3, 3);
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			if (i == 1) {
				kernel(i, j) = 0;
			} else {
				kernel(i, j) = (i==0 ? -1 : 1) * (j==1 ? 2 : 1);
			}
		}
	}

	pop::Mat2F32 out = cpu ? convolution_cpu(in, kernel, 1, 1) : convolution_gpu(in, kernel, 1, 1);
	std::cout << "out = " << out << std::endl; // -> 24
#endif

#if 0
	pop::Mat2UI8 m;
	m.load("/home/pl/workspace/Population/image/Vd-Orig.png");

	pop::Mat2UI8 kernel_identity(3, 3);
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			kernel_identity(i, j) = (i==1 && j==1);
		}
	}

	pop::Mat2UI8 n = cpu ? convolution_cpu(m, kernel_identity, 1, 1) : convolution_gpu(m, kernel_identity, 1, 1);
	n.save("/home/pl/workspace/Population/image/Vd-modified.png");
	std::cout << std::endl;

	n = cpu ? convolution_cpu(m, kernel_identity, 1, 2) : convolution_gpu(m, kernel_identity, 1, 2);
	n.save("/home/pl/workspace/Population/image/Vd-modified-sub.png");
#endif
}

#endif
