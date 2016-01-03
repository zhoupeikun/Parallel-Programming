#include <stdio.h>

#define BLOCK_SIZE    256
#define NUM_ELEMENTS  (4096*100)

// CUDA API error checking macro
#define cudaCheck(error) \
	if (error != cudaSuccess) { \
		printf("Fatal error: %s at %s:%d\n", \
				cudaGetErrorString(error), \
				__FILE__, __LINE__); \
		exit(1); \
	}

__global__ void reverse_1d(int *in, int *out) 
{	
	__shared__ int temp[BLOCK_SIZE];
	int gindex = threadIdx.x + (blockIdx.x * blockDim.x);
	int lindex = threadIdx.x;
	
	temp[BLOCK_SIZE - lindex - 1] = in[gindex];
	
	__syncthreads();	
	
	out[BLOCK_SIZE * (gridDim.x - 1 - blockIdx.x) + threadIdx.x] = temp[lindex];
}

__global__ void reverse_1Drection(int *in, int *out) 
{
	int gindex = threadIdx.x + (blockIdx.x * blockDim.x);
	
	out[NUM_ELEMENTS - gindex - 1] = in[gindex];
}

int main()
{
	unsigned int i;
	int h_in[NUM_ELEMENTS], h_out[NUM_ELEMENTS];
	int *d_in, *d_out;

	// Initialize host data
	for( i = 0; i < (NUM_ELEMENTS); ++i )
		h_in[i] = i;

	// Allocate space on the device
	cudaCheck( cudaMalloc( &d_in, (NUM_ELEMENTS) * sizeof(int)) );
	cudaCheck( cudaMalloc( &d_out, NUM_ELEMENTS * sizeof(int)) );

	// Copy input data to device
	cudaCheck( cudaMemcpy( d_in, h_in, (NUM_ELEMENTS) * sizeof(int), cudaMemcpyHostToDevice) );

	/*cuda timing*/
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	/* end start timing */

	reverse_1d<<< (NUM_ELEMENTS + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE >>> (d_in, d_out);

	/*cuda timing stop*/
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start); cudaEventDestroy(stop);

	cudaCheck( cudaMemcpy( h_out, d_out, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost) );

	// Verify every out value is 7
	for( i = 0; i < NUM_ELEMENTS; ++i )
		if (h_out[i] != h_in[NUM_ELEMENTS-i-1])
		{
			printf("Element h_out[%d] == %d ERROR\n", i, h_out[i]);
			break;
		}

	if (i == NUM_ELEMENTS)
		printf("[SHARED] SUCCESS! %f\n", time);

	// Free out memory
	cudaFree(d_out);

	cudaCheck( cudaMalloc( &d_out, NUM_ELEMENTS * sizeof(int)) );

	/*cuda timing*/
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, 0);
	/* end start timing */

	reverse_1Drection<<< (NUM_ELEMENTS + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE >>> (d_in, d_out);

	/*cuda timing stop*/
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&time, start1, stop1);
	cudaEventDestroy(start1); cudaEventDestroy(stop1);

	cudaCheck( cudaMemcpy( h_out, d_out, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost) );

	// Verify every out value is 7
	for( i = 0; i < NUM_ELEMENTS; ++i )
		if (h_out[i] != h_in[NUM_ELEMENTS-i-1])
		{
			printf("Element h_out[%d] == %d ERROR\n", i, h_out[i]);
			break;
		}

	if (i == NUM_ELEMENTS)
		printf("[PRIVATE] SUCCESS! %f\n", time);

	// Free out memory
	cudaFree(d_in);
	cudaFree(d_out);


	return 0;
}

