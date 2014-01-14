/*
CUDA-accelerated Sauvola's method
For method description see:
Efficient Implementation of Local Adaptive Thresholding Techniques Using Integral Images
http://www.dfki.uni-kl.de/~shafait/papers/Shafait-efficient-binarization-SPIE08.pdf
This code is also a part of paper: SVM-voting Binarization for Degraded Document Images
Submited to ICIP 2014
Author: Xin Chen (xinchen.hawaii@gmail.com) 
Date: November 4, 2013
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <helper_math.h>

// includes, cuda
#include <helper_cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)

#define printf(f, ...) ((void)(f, __VA_ARGS__),0)

#endif

__global__ void 
	binarization_in_Kernal(float *inteImg, float *inteSqImg,		//input
	uchar *data, uchar *outputData,						//input and output for the image to be binarized
	int width, int height, int step, int InterStep, float k, int whalf);	//the image size

//OpenCV header file

//include opencv header file
#include <opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace cv;

//Parameter
#define BLOCK_SIZE 32

//Device
float *d_inteImg;
float *d_inteSqImg;
uchar *d_data;
uchar *d_outputData; //output binary image

#ifndef MAX
#define MAX(a,b) ((a < b) ? b : a)
#endif

#ifndef MIN
#define MIN(a,b)((a > b) ? b : a)
#endif


/*-------------------------------------------------------------------
//data copy from host to device 

inputImg	-	the input image
inteSqImg	-	the integral square image
IntegralNum -   number of intergral image number
*/
extern "C" void dataCopyKernel(Mat &inputImg, float* interImg, float* interSqImg, const long int IntegralNum)
{
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)&d_inteImg,IntegralNum*sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device d_inteImg (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_inteSqImg,IntegralNum*sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device d_inteSqImg (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_data, inputImg.rows * inputImg.step * sizeof(uchar));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device d_data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_outputData, inputImg.rows * inputImg.step * sizeof(uchar));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device d_outputData (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//Copy data from host to device
	checkCudaErrors(cudaMemcpy(d_inteImg,interImg,IntegralNum*sizeof(float),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_inteSqImg,interSqImg,IntegralNum*sizeof(float),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_data,inputImg.data, inputImg.rows * inputImg.step*sizeof(uchar),cudaMemcpyHostToDevice));
}

//Clear memory
extern "C" void memoryClean()
{
	checkCudaErrors(cudaFree(d_inteImg));
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_inteSqImg));
	checkCudaErrors(cudaFree(d_outputData));
}

extern "C" void imageKernel(Mat &input, float coeffient, int integralSize,int whalf)
{
	int imgHeight = input.rows;
	int imgStep = input.step/sizeof(uchar);
	int longstep = MAX(imgStep, input.cols+1); 
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 dimGrid(ceil(1.0f * longstep / BLOCK_SIZE), ceil(1.0f * imgHeight/BLOCK_SIZE));
	
	//CUDA kernel
	binarization_in_Kernal<<<dimGrid,dimBlock>>>(d_inteImg, d_inteSqImg, d_data, d_outputData, input.cols, input.rows, imgStep, imgStep+1, coeffient, whalf);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(input.data,d_outputData,sizeof(uchar)*input.rows * imgStep,cudaMemcpyDeviceToHost));

}


/*-------------------------------------------------------------------
binarization function running in CUDA

inteImg		-	the integral image (width+1) * (height+1)
inteSqImg	-	the integral square image (width+1) * (height+1)
data		-	input for the original image,
width		-	the image's original width
height		-	the image's height
step		-	the image's padding width
k           -   control parameter
whalf       -   half of window
output      -    output for the binarized image   
*/
__global__ void 
	binarization_in_Kernal(float *inteImg, float *inteSqImg,		//input
	uchar *data, uchar *outputData,						//input and output for the image to be binarized
	int width, int height, int step, int InterStep, float k, int whalf)	//the image size
{

	int indexX = blockIdx.x*BLOCK_SIZE + threadIdx.x;// X-axis index
	int indexY =  blockIdx.y*BLOCK_SIZE + threadIdx.y;// X-axis index
		
	int area;
	float diff, sqdiff;
	float diagsum;	//the sum of upper left and lower right corner
	float idiagsum;	//the sum of upper right and lower left corner
	float sqdiagsum;	//the square sum of upper left and lower right corner
	float sqidiagsum;	//the square sum of upper right and lower left corner
	uchar tmpValue;

	if (indexX < width && indexY < height)
	{
		//Get the window concer
		int xmin = MAX(0, indexX - whalf);
		int ymin = MAX(0, indexY - whalf);
		int xmax = MIN(width -1, indexX + whalf);
		int ymax = MIN(height -1, indexY + whalf);
		area = ((xmax-xmin + 1)*(ymax-ymin + 1));

		//	
		int xPlus = xmax++;
		int yPlus = ymax++;
		
		diagsum    = inteImg[xPlus + yPlus * InterStep]  + inteImg[xmin  + (ymin) * InterStep];
		idiagsum   = inteImg[xPlus + (ymin) * InterStep] + inteImg[xmin + yPlus * InterStep];
		diff       = diagsum - idiagsum;
		sqdiagsum  = inteSqImg[xPlus + yPlus * InterStep]  + inteSqImg[xmin  + (ymin ) * InterStep];
		sqidiagsum = inteSqImg[xPlus + (ymin) * InterStep] + inteSqImg[xmin + yPlus * InterStep];
		sqdiff     = sqdiagsum - sqidiagsum;
		float invarea = 1.0f / area;
		////Sauvola's binarization
		float meanWin = diff * invarea;
		float std = sqrtf((float)(sqdiff - (diff * diff) *invarea)/(area-1));
		
		float threshold = meanWin+ meanWin*k*(std*0.0078125-1);
		tmpValue = data[indexY * step + indexX];
		uchar tmp = 255;
		if(tmpValue < threshold)
			tmp = 0;
		
		outputData[indexY * step + indexX] = tmp;


	}//End of imgIndex

}
