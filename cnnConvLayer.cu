// This program executes a typical convolutional layer in regular CNNs
#include <iostream>
#include "cnnConvLayer.h"
#include <stdio.h>
using namespace std;


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

/***	Implement your CUDA Kernel here	***/
__global__
void convLayerGPU()
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;
	int xall = blockDim.x * gridDim.x;
	int yall = blockDim.y * gridDim.y;
	int offset = x + y * xall + z * xall * yall;



}
/***	Implement your CUDA Kernel here	***/

int main()
{
	float convLayerCPUExecTime, convLayerGPUExecTime;
	init();
		


	timespec time_begin, time_end;                                                 
  	clock_gettime(CLOCK_REALTIME, &time_begin);
	convLayerCPU();
  	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "CPU time for executing a typical convolutional layer = " <<  convLayerCPUExecTime / 1000000 << "s" << endl;





 	clock_gettime(CLOCK_REALTIME, &time_begin);

 	dim3 threadPerBlock(1);
 	dim3 numBlocks(512,32,32);

	convLayerGPU<<<numBlocks,threadPerBlock>>>(); 


	cudaDeviceSynchronize(); 
  	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = " << convLayerGPUExecTime / 1000000 << "s" << endl;




	if(checker())
	{
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Sorry! Your result is wrong." << endl;

	ending();
	
	return 0;
}
