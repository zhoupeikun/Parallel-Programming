#include<stdio.h>
#define N (2048*2048)
#define THREADS_NUM 512
#define BLOCK_NUM 32

__global__ void reduction(int *num, int *result)
{
    /* insert code to calculate the index properly using blockIdx.x, blockDim.x, threadIdx.x */
	int index = threadIdx.x + blockIdx.x * blockDim.x;   //global index
	int sum = 0;
	int i;
	for(i = index; i < N; i += THREADS_NUM*BLOCK_NUM)
	{
		sum += num[i]*num[i];	
	}
	result[index] = sum;  //result of index
}

/* experiment with N */
/* how large can it be? */


int main()
{
  	int *data;
	int *gpu_data, *result;  //gpu

	int size = N * sizeof( int );  //data size 
	int sum[THREADS_NUM*BLOCK_NUM];

	/* allocate space for device copies of a, b, c */
	cudaMalloc( (void **) &gpu_data, size );
	cudaMalloc( (void **) &result, size);

	/* allocate space for host copies of a, b, c and setup input values */

	data = (int *)malloc( size );

	for( int i = 0; i < N; i++ )
	{
		data[i] = i;
	}

	/* copy inputs to device */
	/* fix the parameters needed to copy data to the device */
	cudaMemcpy( gpu_data, data, size, cudaMemcpyHostToDevice );

	/* launch the kernel on the GPU */
	/* insert the launch parameters to launch the kernel properly using blocks and threads */ 
	reduction<<<N/THREADS_NUM, THREADS_NUM>>>(gpu_data, result);

	/* copy result back to host */
	/* fix the parameters needed to copy data back to the host */
	cudaMemcpy(&sum, result, sizeof(int)*THREADS_NUM*BLOCK_NUM, cudaMemcpyDeviceToHost );

	//sum of every thread' result
	int final_sum = 0;
	for(int i=0; i < THREADS_NUM*BLOCK_NUM; i++ )
	{
		final_sum += sum[i];
	}

	printf("sum:%d ", final_sum);

	/* clean up */

	free(data);
	cudaFree( gpu_data );
	cudaFree(result);

	return 0;
} /* end main */

