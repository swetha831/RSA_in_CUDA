#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__constant__ long int d_A;
__constant__ long int d_B;
// ***********Device sub-function for checking prime number**************
__device__ int prime(long int A)
{
int i;
int flag=1;
	//int i = threadIdx.x;
	int j = sqrt(float((A)));

	for(i=2;i<=j;i++)
	 {
		if ((A) % i == 0)
		flag=0;
	}
//	printf("A %ld   flag %d\n",A,flag);
	return flag;
}
//***************** Kernel for computing prime ******************************
__global__ void isprime(int * flag)
{
*flag = prime(d_A);

	/*int i = threadIdx.x;
	int j = sqrt(float((d_A)));
	if (i>1 && i <= j && (*flag) == 1) {
		if ((d_A) % i == 0)
			atomicCAS(flag, 1, 0);
	}*/
}

// **********************Kernel To compute d and e for generating key************************

__global__ void ce(long int *t,long int *d_e,long int *d_d)
{
	int my_flag;	
	long int my_id =threadIdx.x+2; // this plus two doesnt have any effect
	long int div=1;
	if(my_id>=2 && my_id<*t)
	{
			if((*t%(my_id))!=0)
		{
			my_flag= prime(my_id);
			//printf("flag%d, my_id %ld d_t %ld\n",my_flag, my_id,*t);
			//printf("my_id %d, *t  %ld, d_A %ld, d_B  %ld flag %d d_e %ld, d_d %ld\n",my_id,*t, d_A, d_B,my_flag,*d_e,*d_d);
			if ((my_flag == 1) && (my_id != d_A) && (my_id != d_B))
			{
				long int k=1; 
				while(div==1) 
				{ 
				    k=k+*t; 

				    if(k%my_id==0)
					{
						div = k/my_id;
					} 
				}
			d_e[my_id] = my_id;
			d_d[my_id] = div;
			//printf("DD %ld DE %ld\n",d_d[my_id],d_e[my_id]);
    			//printf("K %ld my_id %ld div %ld DE %ld",k,my_id,d_d[my_id],d_e[my_id]);
			__syncthreads();
			}
		}
	}
}
//********************** Kernel for generating encrypted message *********************
__constant__ long int d_n;
__global__ void encrypt(long int *d_e,long int *d_msg) 
{ 
	long int pt,k,j,ct;
	//printf("DE in encrypt %ld\n",*d_e);
	pt= (long int) d_msg[threadIdx.x]; 
	pt=pt-96; 
	k=1; 
	for(j=0;j<*d_e;j++) 
	{ 
		k=k*pt; 
		k=k%d_n; 
	} 
	ct=k+96; 
	d_msg[threadIdx.x]=ct;  
 
} 
//********************** Kernel for generating decrypted message *********************
__global__ void decrypt(long int *d_d,long int *d_msg) 
{ 

	long int pt,ct,k,j;  
	ct= (long int)d_msg[threadIdx.x]-96; 
	k=1; 
	for(j=0;j<*d_d;j++) 
	{ 
		k=k*ct; 
		k=k%d_n; 
	} 
	pt=k+96; 
	d_msg[threadIdx.x]= pt;
	//printf(" pt: %ld, %c", pt,pt);  
}
/**
* Host main routine
*/
int
main(void)
{
// Event record defined to capture execution time of the kernel
    cudaEvent_t begin,end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	
	int blocksPerGrid = 0;
	int threadsPerBlock = 0;

	// Allocate the host input prime numbers A, B and flag
	long  int *h_A = (long int *)malloc(sizeof(long int));
	long  int *h_B = (long int *)malloc(sizeof(long int));
	int *flag = (int *)malloc(sizeof(int));
	float isprimeTime;
   
	int * deviceflag = NULL;
	// Verify that allocations succeeded
	if (h_A == NULL)
	{
		fprintf(stderr, "Failed to allocate host data - h_A\n");
		exit(EXIT_FAILURE);
	}
	if (h_B == NULL)
	{
		fprintf(stderr, "Failed to allocate host data - h_B\n");
		exit(EXIT_FAILURE);
	}
	if (flag == NULL)
	{
		fprintf(stderr, "Failed to allocate host data - flag\n");
		exit(EXIT_FAILURE);
	}


	err = cudaMalloc((void **)&deviceflag, sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector flag (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	*flag = 0;
	while (*flag == 0) {
		*flag = 1;
		printf("Enter the first prime number \n");
		scanf("%ld", h_A);
		// Copy the host input prime number A in host memory to the  device memory (used constant memory)
		printf("Copy input data from the host memory to the CUDA device\n");
		err= cudaMemcpyToSymbol(d_A, h_A, sizeof(long int));
		//err = cudaMemcpy(d_A, h_A, sizeof(long int), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy A from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		err = cudaMemcpy(deviceflag, flag, sizeof(int), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy flag from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	
		if(*h_A<1024)	
			blocksPerGrid = 1;
		else
			{
			blocksPerGrid = *h_A/1024;
		if(*h_A%1024>0)
			blocksPerGrid++;
			}
		threadsPerBlock = 1024;
		printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
		// Launch the kernel for checking prime number
		cudaEventRecord(begin,0);
		isprime << <blocksPerGrid, threadsPerBlock >> >(deviceflag);

		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch isprime kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// Copy the device result flag in device memory to the host flag
		// in host memory.
		printf("Copy output data from the CUDA device to the host memory\n");
		err = cudaMemcpy(flag, deviceflag, sizeof(int), cudaMemcpyDeviceToHost);// FILL HERE

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	//	printf("FLAg %d\n", *flag);
	}
	// Repeat for prime number B
	*flag = 0;
	while (*flag == 0) {
		*flag = 1;

		err = cudaMemcpy(deviceflag, flag, sizeof(int), cudaMemcpyHostToDevice);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy flag from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		printf("Enter the second prime number \n");
		scanf("%ld", h_B);
		err= cudaMemcpyToSymbol(d_A, h_B, sizeof(long int));
		//err = cudaMemcpy(d_B, h_B, sizeof(long int), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy B from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		if(*h_B<1024)	
			blocksPerGrid = 1;
		else
			{
			blocksPerGrid = *h_A/1024;
		if(*h_B%1024>0)
			blocksPerGrid++;
			}
		threadsPerBlock = 1024;
		// Launch the isprime kernel again for second number
		isprime <<<blocksPerGrid, threadsPerBlock >>>(deviceflag);
printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
		printf("Copy output data from the CUDA device to the host memory\n");
		err = cudaMemcpy(flag, deviceflag, sizeof(int), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy flag from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	//	printf("Flag after %d\n", *flag);
	}
		err= cudaMemcpyToSymbol(d_A, h_A, sizeof(long int));
		//err = cudaMemcpy(d_A, h_A, sizeof(long int), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy A from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		err= cudaMemcpyToSymbol(d_B, h_B, sizeof(long int));
		//err = cudaMemcpy(d_A, h_A, sizeof(long int), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy A from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	// Input the message aftre checking prime number
	char msg[100];
	long int i,h_n;
	long int m[100];
	long int *d_t = NULL;
	long int *d_e = NULL;
	long int *d_d = NULL;
	long int* d_msg = NULL;
	printf("\nENTER MESSAGE\n");
	fflush(stdin);
	scanf("%s", msg);
	for (i = 0; msg[i] != NULL; i++)
	{
		m[i] = msg[i];
	}
	printf("Message: %s\n", msg);


	err = cudaMalloc((void **)&d_msg,strlen(msg)* sizeof(long int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector flag (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_msg, &m, strlen(msg)* sizeof(long int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Comnpute N which is A*B
	h_n = (*h_A)*(*h_B);

		err= cudaMemcpyToSymbol(d_n, &h_n, sizeof(long int));
	    if (err != cudaSuccess)
	    {
		fprintf(stderr, "Failed to copy matrix A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	    }	
	long  int *h_t = (long int *)malloc(sizeof(long int));

	if (h_t == NULL)
	{
		fprintf(stderr, "Failed to allocate host t!\n");
		exit(EXIT_FAILURE);
	}
// Calculate totient function
	*h_t = (*h_A - 1)*(*h_B - 1);
	err = cudaMalloc((void **)&d_t, sizeof(long int));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_t,h_t, sizeof(long int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	long  int *h_e = (long int *)malloc((*h_t)*sizeof(long int));

	if (h_e == NULL)
	{
		fprintf(stderr, "Failed to allocate host h_e!\n");
		exit(EXIT_FAILURE);
	}
	for (int iterator=0; iterator<*h_t;iterator++){
		h_e[iterator]=0;
	}
	err = cudaMalloc((void **)&d_e, (*h_t)*sizeof(long int));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_e (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_e, h_e, (*h_t) * sizeof(long int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector d_e from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	long  int *h_d = (long int *)malloc((*h_t)*sizeof(long int));

	if (h_d == NULL)
	{
		fprintf(stderr, "Failed to allocate host h_d!\n");
		exit(EXIT_FAILURE);
	}


	for (int iterator=0; iterator<*h_t;iterator++){
		h_d[iterator]=0;
	}
	err = cudaMalloc((void **)&d_d, (*h_t)*sizeof(long int));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_d (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_d, h_d, (*h_t) * sizeof(long int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector d_e from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	if(*h_t<1024)
		blocksPerGrid = 1;
	else
		{
		blocksPerGrid = *h_t/1024;
	if(*h_t%1024>0)
		blocksPerGrid++;
		}
	threadsPerBlock = 1024;
printf("CUDA CE launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
// Kernel launch for CE compoutation which will generate approrpiate d and e values 
	ce << <1, threadsPerBlock >> > (d_t,d_e,d_d);

	blocksPerGrid = 1;
	threadsPerBlock = strlen(msg);
	err = cudaMemcpy(h_e, d_e, (*h_t)*sizeof(long int), cudaMemcpyDeviceToHost);// FILL HERE

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector h_e from device to host (error code %s)!\n", cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}


	err = cudaMemcpy(h_d, d_d, (*h_t)*sizeof(long int), cudaMemcpyDeviceToHost);// FILL HERE

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector h_d from device to host (error code %s)!\n", cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}
	// Free device global memory d_e, d_d and assign one element in device for d_e, d_d
	err = cudaFree(d_e);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device d_e (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	err = cudaMalloc((void **)&d_e, sizeof(long int));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_e (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_d);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device d_d (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	err = cudaMalloc((void **)&d_d, sizeof(long int));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_d (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int iterator=0; iterator<*h_t;iterator++)
	{
		if(h_e[iterator]!=0)//&& (h_e[iterator] != h_d[iterator]))
		{
			printf("%d : h_e  %ld  h_d  %ld \n",iterator, h_e[iterator],h_d[iterator]);
			err = cudaMemcpy(d_e, &h_e[iterator], sizeof(long int), cudaMemcpyHostToDevice);
			if (err != cudaSuccess)
			{
				fprintf(stderr, "Failed to copy vector d_e from device to host (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
			err = cudaMemcpy(d_d, &h_d[iterator], sizeof(long int), cudaMemcpyHostToDevice);
			if (err != cudaSuccess)
			{
				fprintf(stderr, "Failed to copy vector d_d from device to host (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
		break;
		}
	}

printf("CUDA encrypt launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
// Kernel Launch to generate cipher text
	encrypt<<<blocksPerGrid, threadsPerBlock >>>(d_e, d_msg);


	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(&m, d_msg, strlen(msg)*sizeof(long int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("\n\nEncrypted message is: ");
	for (i = 0;i<strlen(msg); i++)
	{
		printf("%c",m[i]);
	}
	printf("\n");
// Kernel launch for decrypting the message with the approrpiate key
	decrypt<<<blocksPerGrid, threadsPerBlock >>>(d_d, d_msg);


	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(&m, d_msg, strlen(msg)*sizeof(long int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("\n\nDecrypted message is: ");
	for (i = 0;i<strlen(msg); i++)
	{
		printf("%c",m[i]);
	}
	printf("\n");
cudaEventRecord(end,0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&isprimeTime, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
printf("Total Time: %f millisecond \n",isprimeTime);
	/*err = cudaFree(d_A);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_B);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}*/


	err = cudaFree(deviceflag);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	free(h_A);
	free(h_B);
	free(flag);

	// Reset the device and exit
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	err = cudaDeviceReset();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Done\n");
	return 0;
}

