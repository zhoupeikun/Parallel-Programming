#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>

_global_ void vector_inverse(int *a, int *b, int N)
{
    int index = threadIdx.x + blockIdx * blockDim.x;
    int tmp = a[index];
    b[index] = a[N-index-1];
    b[N-index-1] = tmp;
}

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

int main()
{
    int *a, *b;
    int *d_a, *d_b;

    int size = N * sizeof(int);

    //allocate space for device copies of a
    cudaMalloc((void **) &d_a, size);

    //allocte space for host of a and setup input values

    a = (int *)malloc( size);

    for(int i = 0; i < N; i++)
    {
    a[i] = i;
    }

    //copy inputs to device
    cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice);

	struct timeval tim;
	gettimeofday(&tim, NULL);
	double t1 = tim.tv_sec + (tim.tv_usec/1000000.0);

    //launch the kernel on the GPU
    vector_inverse<<<N/THREAD_PER_BLOCK, THREADS_PER_BLOCKS >>>(d_a, N);

    //copy result back to host
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

	gettimeofday(&tim, NULL);
	double t2 = tim.tv_sec + (tim.tv_usec/100000.0);

	printf("Temps total = %f sec\n", t2 - t1);

    free(a);
    cudaFree(d_a);

    return 0;
}