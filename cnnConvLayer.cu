// This program executes a typical convolutional layer in regular CNNs
#include <iostream>
#include "cnnConvLayer.h"
#include <stdio.h>
using namespace std;

#define xDim 512
#define yDim 32
#define zDim 32

#define xThreadDim 4
#define yThreadDim 16
#define zThreadDim 16

int *devOut;
int outputsize = xDim * yDim * zDim;
int *outResult = new int[outputsize]();

int *CPUout = new int[outputsize](); //

// This is the CPU version, please don't modify it
void convLayerCPU()
{
	// declarations for bunch of indexing parameters
	int fn, sli, fmy, fmx, y, x;
	int sum, ifmy, ifmx, ofmy, ofmx;
	int filtIdx, inNeuIdx, outNeuIdx, outIdx;
	int filtVol = FMDEPTH * FILTSIZE * FILTSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int fmArea = FMSIZE *FMSIZE;
	int outArea = FMSIZE/2 * FMSIZE/2;


	cout << "convolutioning..." << endl;

	// Convolution
	for(fn = 0; fn < FILTNUM; fn++) //512
	{
		for(fmy = 0; fmy < FMSIZE; fmy += STRIDE) //32
		{
			for(fmx = 0; fmx < FMSIZE; fmx += STRIDE)  //32
			{
				sum = 0;
				for(sli = 0; sli < FMDEPTH; sli++)  //512
				{
					for(y = 0; y < FILTSIZE; y++)  //3
					{
						for(x = 0; x < FILTSIZE; x++)  //3
						{
							ifmy = fmy - FILTSIZE / 2 + y;		//no dependancy
							ifmx = fmx - FILTSIZE / 2 + x;		//no dependancy
							filtIdx = (fn * filtVol) + (sli * filtArea) + (y * FILTSIZE) + x;	//no dependancy
							inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;							//no dependancy
							if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
								sum += filt[filtIdx] * inNeu[inNeuIdx];
						}
					}
				}
				// Activation - ReLU
				outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
				if(sum <= 0)
					outNeu[outNeuIdx] = 0;
				else
					outNeu[outNeuIdx] = sum;
			}
		}
	}


 	cout << "Pooling....." << endl;
	// Max Pooling with Window Size 2x2
	int max, tmpVal;
	for(sli = 0; sli < FILTNUM; sli++)
	{
		for(fmy = 0; fmy < FMSIZE/2 ; fmy += 1)
		{
			for(fmx = 0; fmx < FMSIZE/2 ; fmx += 1)
			{
				outNeuIdx = sli*fmArea + fmy*2*FMSIZE + fmx*2;
				max = outNeu[outNeuIdx];
				for(y = 0; y < 2; y++)
				{
					for(x = 0; x < 2; x++)
					{
						ofmy = fmy*2 + y;
						ofmx = fmx*2 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = outNeu[outNeuIdx];
						if(tmpVal > max)
							max = tmpVal;
					}
				}
				outIdx = sli*outArea + fmy*FMSIZE/2 + fmx;
				outCPU[outIdx] = max;
			}
		}
	}
}


void initGPU()
{
	cudaMalloc(&devOut, sizeof(int)*outputsize);
}

void CPUrun()
{
	int i;

	for (i = 0; i < outputsize; ++i)
	{
		CPUout[i] = 1;
	}
}


/***	Implement your CUDA Kernel here	***/
__global__
void convLayerGPU(int *out)
{
	int threadX = threadIdx.x + blockIdx.x * blockDim.x;
	int threadY = threadIdx.y + blockIdx.y * blockDim.y;
	int threadZ = threadIdx.z + blockIdx.z * blockDim.z;
	int xall = blockDim.x * gridDim.x;
	int yall = blockDim.y * gridDim.y;
	int GlobalThreadId = threadX + threadY * xall + threadZ * xall * yall;
	int GlobalBlockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;

	if (GlobalBlockId == 1)
		out[GlobalThreadId] = 1;
	else
		out[GlobalThreadId] = 0;

}
/***	Implement your CUDA Kernel here	***/

int main()
{
	float convLayerCPUExecTime, convLayerGPUExecTime;
	init();



	timespec time_begin, time_end;
  	clock_gettime(CLOCK_REALTIME, &time_begin);
	//convLayerCPU();
	CPUrun();
  	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << " ================ Result ===================" << endl;
	cout << "CPU time for executing a typical convolutional layer = " <<  convLayerCPUExecTime / 1000 << "ms" << endl;





 	initGPU();
 	dim3 threadPerBlock(xThreadDim, yThreadDim, zThreadDim);
 	dim3 numBlocks(xDim/xThreadDim, yDim/yThreadDim, zDim/zThreadDim);
 	clock_gettime(CLOCK_REALTIME, &time_begin);


	convLayerGPU<<<numBlocks,threadPerBlock>>>(devOut);


	cudaDeviceSynchronize();
  	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = " << convLayerGPUExecTime / 1000 << "ms" << endl;


	int outSize = sizeof(int)*outputsize;
	cudaMemcpy(outResult, devOut, outSize, cudaMemcpyDeviceToHost);
	cudaFree(&devOut);


	int sumGPU = 0;
	//int sumCPU = 0;
	for (int i = 0; i < outputsize; ++i)
	{
		if (outResult[i] == 1)
		{
			printf("%d  ", i);
			sumGPU++;
		}
		//printf("%d  ",outResult[i] );
		//sumGPU += outResult[i];
		//sumCPU += CPUout[i];
	}
	printf("sumGPU = %d\n",sumGPU );

	/*if((sumCPU - sumGPU) == 0)
		printf("right\n");
	else
		printf("wrong\n");*/

	delete [] outResult;
	delete [] CPUout;



/*

	if(checker())
	{
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Sorry! Your result is wrong." << endl;
*/
	ending();

	return 0;
}
