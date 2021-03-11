#include <stdio.h>
#include "mj-draw.h"
#include <jetson-utils/cudaUtility.h>

int min_of_them(int a, int b){
	return a < b ? a : b;
} 

__global__ void gpuDrawCross_mj (uchar3 * img, int width, int height, int crossLen, int thickness)
{
	uchar3 pixel_temp;
	pixel_temp.x = 0;
	pixel_temp.y = 250;
	pixel_temp.z = 0;

	if(thickness > 8
	|| thickness < 0){
		thickness = 8;
	}

    int box_x = blockIdx.x * blockDim.x + threadIdx.x;
	int box_y = blockIdx.y * blockDim.y + threadIdx.y;

	int img_x = (width - crossLen) / 2 + box_x;
	int img_y = (height - crossLen) / 2 + box_y;

	if(img_x > width
	|| img_y > height){
		return;
	}
	
	uint i = 0;
	if(box_y == (crossLen / 2)){
		for(i = 0;i < thickness;++i){
			img[ (img_y - i/2) * width + img_x] = pixel_temp;		
		}

	}
	if(box_x == (crossLen / 2)){
		for(i = 0;i < thickness;++i){
			img[ img_y * width + (img_x - i/2) ] = pixel_temp;	
		}

	}
}



void mj_draw_test(uchar3* img, int width, int height, int cross_len, int thickness)
{
	if(cross_len > width
	|| cross_len > height){

		cross_len = min_of_them(width, height);
	} 

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(cross_len, blockDim.x), iDivUp(cross_len, blockDim.y));

    gpuDrawCross_mj<<<gridDim, blockDim>>>( img, width, height, cross_len, thickness);
}
