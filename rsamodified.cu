#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define WIDTH 1024
//Constant memeory use for read only values
__constant__ long int d_A;
__constant__ long int d_B;
__constant__ long int d_n;
// ***********Device sub-function for checking prime number**************
__device__ int prime(long int A)
{
	int i;
	int flag=1;
	int j = sqrtf(A);
	for(i=2;i<=j;i++)
	 {
		if ((A) % i == 0)
		flag=0;
	}
	return flag;
}
// **********************Child Kernel To compute d and e for generating key************************
__global__ void ce(long int *d_t,long int* d_e, long int* d_d)
{
	int my_flag,j;	
	long int my_id =threadIdx.x;
	long int div=1;
// Shared mememory usage for fast intermediate calculation
	__shared__ long int s_e[WIDTH];
	__shared__ long int s_d[WIDTH];
	if(my_id>=2 && my_id<*d_t)
	{
		if((*d_t%(my_id))!=0)
		{
			my_flag= prime(my_id);
			if ((my_flag == 1) && (my_id != d_A) && (my_id != d_B))
			{
				long int k=1; 
				while(div==1) 
				{ 
				    k=k+*d_t; 

				    if(k%my_id==0)
					{
						div = k/my_id;
					} 
				}
			s_e[my_id] = my_id;
			s_d[my_id] = div;
			__syncthreads();
			}
		}
	
	}
__syncthreads();
	if(threadIdx.x==0)
	{
	for(j=0;j<blockDim.x;j++)
	{
		if(s_e[j] !=0)
		{	
			*d_e=s_e[j];
			*d_d=s_d[j];
			break;
		}
	}

	}
	
}
//**********************Child Kernel for generating encrypted message *********************
__global__ void encrypt(long int *d_e,long int *d_msg,long int *e_msg) 
{
	__shared__ long int msg[WIDTH];
	msg[threadIdx.x]=d_msg[blockDim.x*blockIdx.x+threadIdx.x]; 
	long int pt,k,j,ct;
	pt= msg[threadIdx.x]; 
	pt=pt-96; 
	k=1; 
	for(j=0;j<*d_e;j++) 
	{ 
		k=k*pt; 
		k=k%d_n; 
	} 
	ct=k+96; 
	msg[threadIdx.x]=ct;
	e_msg[blockDim.x*blockIdx.x+threadIdx.x]=msg[threadIdx.x];
	__syncthreads(); 
} 
//**********************Child Kernel for generating decrypted message *********************
__global__ void decrypt(long int *d_d,long int *d_msg,long int* e_msg) 
{ 
	__shared__ long int msg[WIDTH];
	msg[threadIdx.x]=e_msg[blockDim.x*blockIdx.x+threadIdx.x];
	long int pt,ct,k,j;  
	ct=msg[threadIdx.x]-96; 
	k=1; 
	for(j=0;j<*d_d;j++) 
	{ 
		k=k*ct; 
		k=k%d_n; 
	} 
	pt=k+96; 
	msg[threadIdx.x]= pt; 
	d_msg[blockDim.x*blockIdx.x+threadIdx.x]=msg[threadIdx.x];
	__syncthreads(); 
}
// Main parent kernel 
__global__ void rsa(long int* d_t,long int* d_e, long int* d_d,long int* d_msg,long int *e_msg,int * d_len)
{
	int blocksPerGrid,threadsPerBlock;
	int flag1,flag2;
	if(threadIdx.x==0)
	{
		flag1= prime(d_A);
		flag2= prime(d_B);

		if(flag1==1 && flag2 ==1)
		{
			*d_t = __mul24((d_A-1),(d_B-1));
//	printf("d_t = %ld", *d_t);
			if(*d_t<1024)
				blocksPerGrid = 1;
			else
				{
				blocksPerGrid = *d_t/1024;
			if(*d_t%1024>0)
				blocksPerGrid++;
				}
			threadsPerBlock = 1024;
		printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
// Launch the first child kernel
			ce << <blocksPerGrid, threadsPerBlock>> > (d_t,d_e,d_d);
			cudaDeviceSynchronize();
			__syncthreads(); 
			if(*d_len<1024)
				blocksPerGrid = 1;
			else
				{
				blocksPerGrid = *d_len/1024;
				if(*d_len%1024>0)
					blocksPerGrid++;
				}
			threadsPerBlock = 1024;
			printf("Encrypting...\n");

// launch the second child kernel 
			encrypt << <blocksPerGrid, threadsPerBlock>> > (d_e,d_msg,e_msg);	
			cudaDeviceSynchronize();
			__syncthreads(); 
			printf("\n\nEncrypted message is: ");
			for (int i = 0;i<*d_len; i++)
			{
				printf("%c",e_msg[i]);
			}
			printf("\n"); 
			printf("Decrypting...\n");
			printf("CUDA kernel launch for encryption and decryption with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
// Launch the third child kernel
			decrypt << <blocksPerGrid, threadsPerBlock>> > (d_d,d_msg,e_msg);		
			cudaDeviceSynchronize();
			__syncthreads();

			printf("\n\nDecrypted message is: ");
			for (int i = 0;i<*d_len; i++)
			{
				printf("%c",d_msg[i]);
			}
			printf("\n");
		}
		else
			printf("Wrong Input !!Enter correct prime numbers to proceed. \n");
	//printf("DE %ld DD %ld\n",*d_e,*d_d);	
	}
}
// Host main code
int main(void)
{
// Cuda event api to record executiion time
cudaEvent_t begin,end;
cudaEventCreate(&begin);
cudaEventCreate(&end);
cudaError_t err = cudaSuccess;
long  int *h_A = (long int *)malloc(sizeof(long int));
long  int *h_B = (long int *)malloc(sizeof(long int));
int *flag = (int *)malloc(sizeof(int));
char msg[10*WIDTH];
long int m[10*WIDTH];
long int d[10*WIDTH];
long int h_n;
long int *d_e = NULL;
long int *d_d = NULL;
long  int *h_e = (long int *)malloc(sizeof(long int));
long  int *h_d = (long int *)malloc(sizeof(long int));

long int *d_t= NULL;
long  int *h_t = (long int *)malloc(sizeof(long int));
*h_e=0;
*h_d=0;
float isprimeTime;
int len=0;
int *d_len = NULL;
err = cudaMalloc((void **)&d_len, sizeof(int));

if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to allocate device vector d_len (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}
err = cudaMalloc((void **)&d_t, sizeof(long int));

if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to allocate device vector d_e (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}

err = cudaMemcpy(d_t, h_t,sizeof(long int), cudaMemcpyHostToDevice);
if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to copy vector d_t from device to host (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}

err = cudaMalloc((void **)&d_e, 99);

if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to allocate device vector d_e (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}

err = cudaMemcpy(d_e, h_e,sizeof(long int), cudaMemcpyHostToDevice);
if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to copy vector d_e from device to host (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}

err = cudaMalloc((void **)&d_d, sizeof(long int));

if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to allocate device vector d_d (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}

err = cudaMemcpy(d_d, h_d,sizeof(long int), cudaMemcpyHostToDevice);
if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to copy vector d_d from device to host (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}


long int* d_msg = NULL;
long int* e_msg = NULL;
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

printf("Enter the first prime number \n");
		scanf("%ld", h_A);
printf("Copy input data from the host memory to the CUDA device\n");
err= cudaMemcpyToSymbol(d_A, h_A, sizeof(long int));

if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to copy A from host to device (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}
printf("Enter the second prime number \n");
		scanf("%ld", h_B);
		err= cudaMemcpyToSymbol(d_B, h_B, sizeof(long int));

printf("\nENTER MESSAGE\n");
fflush(stdin);
scanf("%s", msg);
int i=0;
for (i = 0; msg[i] != NULL; i++)
{
	m[i] = msg[i];
}
//printf("Message: %s\n", msg);
len = strlen(msg);
err = cudaMemcpy(d_len, &len,sizeof(int), cudaMemcpyHostToDevice);
if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to copy vector d_len from device to host (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}
err = cudaMalloc((void **)&d_msg,len* sizeof(long int));
if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to allocate d_len flag (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}
err = cudaMalloc((void **)&e_msg,len* sizeof(long int));
if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to allocate device vector flag (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}
err = cudaMemcpy(d_msg, &m, len* sizeof(long int), cudaMemcpyHostToDevice);
if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to copy length from host to device (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}
err = cudaMemcpy(e_msg, &m,len* sizeof(long int), cudaMemcpyHostToDevice);
if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to alocate for encrypted message from host to device (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}
h_n = (*h_A)*(*h_B);

err= cudaMemcpyToSymbol(d_n, &h_n, sizeof(long int));
if (err != cudaSuccess)
{
fprintf(stderr, "Failed to copy matrix N from host to device (error code %s)!\n", cudaGetErrorString(err));
exit(EXIT_FAILURE);
}	
// Begin the main kernel launch
cudaEventRecord(begin,0);
rsa <<< 1,1>>>(d_t,d_e,d_d,d_msg,e_msg,d_len);

err = cudaGetLastError();
if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}
/*err = cudaMemcpy(&m, e_msg, strlen(msg)*sizeof(long int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("Encrypted message is: ");
	for (i = 0;i<strlen(msg); i++)
	{
		printf("%c",m[i]);
	}
	printf("\n");



err = cudaMemcpy(&d, d_msg, strlen(msg)*sizeof(long int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("Decrypted message is: ");
	for (i = 0;i<strlen(msg); i++)
	{
		printf("%c",d[i]);
	}
	printf("\n");
*/
cudaEventRecord(end,0);
cudaEventSynchronize(end);
cudaEventElapsedTime(&isprimeTime, begin, end);
cudaEventDestroy(begin);
cudaEventDestroy(end);
printf("Total Time: %9f millisecond \n",isprimeTime);
free(h_A);
free(h_B);
err = cudaDeviceReset();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Done\n");
	return 0;
}

