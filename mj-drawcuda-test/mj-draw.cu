#include <stdio.h>
#include "mj-draw.h"
#include <jetson-utils/cudaUtility.h>

int min_of_them(int a, int b){
	return a < b ? a : b;
} 

__global__ void gpuDrawBox(uchar3 * img, int width, int height, int box_width, int box_height, int thickness)
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

	int img_x = (width - box_width)/2 + box_x;
	int img_y = (height - box_height)/2 + box_y;

	if((img_x + thickness/2) >= width
	|| (img_y + thickness/2) >= height){
		return;
	}

	if(box_x >= box_width
	|| box_y >= box_height){
		return;
	}

	int i = 0;
	if(box_x == 0
	|| box_x == (box_width - 1))
	{
		for(i = 0;i < thickness;++i){
			img[img_y * width + (img_x - thickness/2 + i)] = pixel_temp;
		}
	}
	if(box_y == 0 
	|| box_y == (box_height - 1))
	{	
		for(i = 0;i < thickness;++i){
			img[(img_y - thickness/2 + i) * width + img_x] = pixel_temp;
		}
	}

}

/*
* draw a cross, input para "int2 pos " is the origin pos of box, not the center pos of box
*/
__global__ void gpuDrawCross_pos (uchar3 * img, int width, int height, int crossLen, int thickness, int2 pos)
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

	int img_x = pos.x + box_x;
	int img_y = pos.y + box_y;

	if(img_x >= width
	|| img_y >= height){
		return;
	}

	if(box_x >= crossLen
	|| box_y >= crossLen){
		return;
	}
	
	uint i = 0;
	if(box_y == (crossLen / 2)){
		for(i = 0;i < thickness;++i){
			img[ (img_y - thickness/2 + i) * width + img_x] = pixel_temp;		
		}

	}
	if(box_x == (crossLen / 2)){
		for(i = 0;i < thickness;++i){
			img[ img_y * width + (img_x - thickness/2 + i) ] = pixel_temp;	
		}

	}
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

	if(img_x >= width
	|| img_y >= height){
		return;
	}

	if(box_x >= crossLen
	|| box_y >= crossLen){
		return;
	}
	
	uint i = 0;
	if(box_y == (crossLen / 2)){
		for(i = 0;i < thickness;++i){
			img[ (img_y - thickness/2 + i) * width + img_x] = pixel_temp;		
		}

	}
	if(box_x == (crossLen / 2)){
		for(i = 0;i < thickness;++i){
			img[ img_y * width + (img_x - thickness/2 + i)] = pixel_temp;	
		}

	}
}


/************
*
*	call
* function
*
*************/

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

void mj_draw_test_pos(uchar3* img, int width, int height, int cross_len, int thickness, int2 pos)
{
	if(cross_len > width
	|| cross_len > height){

		cross_len = min_of_them(width, height);
	} 

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(cross_len, blockDim.x), iDivUp(cross_len, blockDim.y));

    gpuDrawCross_pos<<<gridDim, blockDim>>>( img, width, height, cross_len, thickness, pos);
}

void mj_drawBox_test(uchar3* img, int width, int height, int box_w, int box_h, int thickness)
{
	if(box_w > width
	|| box_h > height){
		return ;
	} 

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(box_w, blockDim.x), iDivUp(box_h, blockDim.y));

    gpuDrawBox<<<gridDim, blockDim>>>( img, width, height, box_w, box_h, thickness);
}