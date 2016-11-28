// This program executes a typical convolutional layer in regular CNNs
#include <iostream>
#include "cnnConvLayer.h"
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

	// Convolution
	for(fn = 0; fn < FILTNUM; fn++){
		for(fmy = 0; fmy < FMSIZE; fmy += STRIDE){
			for(fmx = 0; fmx < FMSIZE; fmx += STRIDE){
				for(sli = 0; sli < FMDEPTH; sli++){
					sum = 0;
					for(y = 0; y < FILTSIZE; y++){
						for(x = 0; x < FILTSIZE; x++){
							ifmy = fmy - FILTSIZE / 2 + y;
							ifmx = fmx - FILTSIZE / 2 + x;
							filtIdx = fn*filtVol + sli*filtArea + y*FILTSIZE + x;
							inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
							if(ifmy > 0 && ifmy < FMSIZE && ifmx > 0 && ifmx < FMSIZE)
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

	// Max Pooling with Window Size 2x2
	int max, tmpVal;
	for(sli = 0; sli < FILTNUM; sli++){
		for(fmy = 0; fmy < FMSIZE/2 ; fmy += 1){
			for(fmx = 0; fmx < FMSIZE/2 ; fmx += 1){
				outNeuIdx = sli*fmArea + fmy*2*FMSIZE + fmx*2;
				max = outNeu[outNeuIdx];
				for(y = 0; y < 2; y++){
					for(x = 0; x < 2; x++){
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
}
/***	Implement your CUDA Kernel here	***/

int main()
{
	int convLayerCPUExecTime, convLayerGPUExecTime;
	init();
		
	timespec time_begin, time_end;                                                 
  clock_gettime(CLOCK_REALTIME, &time_begin);

	convLayerCPU();

  clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "CPU time for executing a typical convolutional layer = " 
			 <<  convLayerCPUExecTime / 1000 << "ms" << endl;

  clock_gettime(CLOCK_REALTIME, &time_begin);
	/***	Lunch your CUDA Kernel here	***/

	convLayerGPU<<<1,1>>>(); // Lunch the kernel
	
	cudaDeviceSynchronize(); // Do synchronization before clock_gettime()
	/***	Lunch your CUDA Kernel here	***/
  clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = " 
			 << convLayerGPUExecTime / 1000 << "ms" << endl;

	if(checker()){
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Sorry! Your result is wrong." << endl;

	ending();
	
	return 0;
}
