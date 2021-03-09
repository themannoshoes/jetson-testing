#include <stdio.h>
#include "mj-draw.h"

int min_of_them(int a, int b){
	return a < b ? a : b;
} 

__global__ void gpudrawX_mj (uchar3 * img, int width, int height, int n)
{
	uchar3 pixel_temp;
	pixel_temp.x = 0;
	pixel_temp.y = 250;
	pixel_temp.z = 0;

    int i = threadIdx.x;
	int head_x_axis = (width - n )/2;
    if(i < n){
		img[height/2 * width + (head_x_axis + i)] = pixel_temp;

		//two more code below is to add thickness 
		img[(height/2 -1)* width + (head_x_axis + i)] = pixel_temp;
		img[(height/2 +1)* width + (head_x_axis + i)] = pixel_temp;
	}
}

__global__ void gpudrawY_mj (uchar3 * img, int width, int height, int n)
{
	uchar3 pixel_temp;
	pixel_temp.x = 0;
	pixel_temp.y = 250;
	pixel_temp.z = 0;

    int i = threadIdx.x;
	int head_y_axis = (height - n) /2;

	if(i < n){
		img[(i + head_y_axis) * width + width/2] = pixel_temp;

		//two more code below is to add thickness 
		img[(i + head_y_axis) * width + width/2 -1] = pixel_temp;
		img[(i + head_y_axis) * width + width/2 + 1] = pixel_temp;
	}
}

void mj_draw_test(uchar3* img, int width, int height, int cross_len)
{
	if(cross_len > width
	|| cross_len > height){

		cross_len = min_of_them(width, height);
	} 
//	cross_len = 200;
    gpudrawX_mj<<<1, cross_len>>>( img, width, height, cross_len);
	gpudrawY_mj<<<1, cross_len>>>( img, width, height, cross_len);
}
