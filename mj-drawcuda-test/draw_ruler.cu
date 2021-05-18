#include "mj-draw.h"
#include <jetson-utils/cudaFont.h>
#include <float.h>
#include <math.h>


/*
* draw a graduated ruler, the orgin position is parameter "int2 pos"
*/
__global__ void gpuDrawLevelRuler (uchar3 * img, int width, int height, int ruler_len, int ruler_tooth_height, int thickness, int2 pos)
{
	uchar3 pixel_temp;
	pixel_temp.x = 0;
	pixel_temp.y = 250;
	pixel_temp.z = 0;


	if(thickness > (ruler_tooth_height /4)
	|| thickness < 0){
		thickness = ruler_tooth_height /4;
	}

    int box_x = blockIdx.x * blockDim.x + threadIdx.x;
	int box_y = blockIdx.y * blockDim.y + threadIdx.y;

	int img_x = pos.x + box_x;
	int img_y = pos.y + box_y;

	if(img_x >= width
	|| img_y >= height){
		return;
	}

	if(box_x >= ruler_len
	|| box_y >= ruler_tooth_height ){
		return;
	}
	
	int tooth_num = 12;
	float tooth_gap = (float)ruler_len / tooth_num;
	int i;

	if(box_y == (ruler_tooth_height - thickness /2) ){
		for(i = 0; i < thickness;i++){
			img[ (img_y + i - thickness/2) * width + img_x] = pixel_temp;
		}
	}
	
	float mod,temp_f;
	int temp_int;
	temp_f = box_x / tooth_gap;
	temp_int = temp_f;
	mod = box_x - temp_int * tooth_gap;
    if(mod < 0)mod = -mod;
	if( mod < 1
	|| box_x == ruler_len -1){
		for(i = 0;i < thickness;i++){
			img[img_y * width + (img_x + i - thickness/2)] = pixel_temp;
		}
	}
}

/*
description: draw a vertical graduated ruler
para:
*/
__global__ void gpuDrawVerticalRuler (uchar3 * img, int width, int height, int ruler_len, int ruler_tooth_height, int thickness, int2 pos)
{
	uchar3 pixel_temp;
	pixel_temp.x = 0;
	pixel_temp.y = 250;
	pixel_temp.z = 0;


	if(thickness > (ruler_tooth_height /4)
	|| thickness < 0){
		thickness = ruler_tooth_height /4;
	}

    int box_x = blockIdx.x * blockDim.x + threadIdx.x;
	int box_y = blockIdx.y * blockDim.y + threadIdx.y;

	int img_x = pos.x + box_x;
	int img_y = pos.y + box_y;

	if(img_x >= width
	|| img_y >= height){
		return;
	}

	if(box_x >= ruler_tooth_height
	|| box_y >= ruler_len ){
		return;
	}
	
	int tooth_num = 12;
	float tooth_gap = (float)ruler_len / tooth_num;
	int i;

	if(box_x == (ruler_tooth_height - thickness/2)){
		for(i = 0;i < thickness;i++){
			img[ img_y * width + (img_x + i - thickness/2)] = pixel_temp;
		}
	}
	
	float mod,temp_f;
	int temp_int;
	temp_f = box_y / tooth_gap;
	temp_int = temp_f;
	mod = box_y - temp_int * tooth_gap;
    if(mod < 0)mod = -mod;
	if( mod < 1
	|| box_y == ruler_len -1){
		for(i = 0;i < thickness;i++){
			img[(img_y + i - thickness/2) * width + img_x] = pixel_temp;
		}
	}
}



/*
description: draw a line base on two point we give
*/
__global__ void gpuDrawStraightLines(uchar3 * img, int width, int height, int thickness, int2 pos1, int2 pos2)
{
	uchar3 pixel_temp;
	pixel_temp.x = 0;
	pixel_temp.y = 250;
	pixel_temp.z = 0;

	//get the box width and box height	
	int box_width = pos1.x - pos2.x;
	if(box_width < 0)box_width *= -1;

	int box_height = pos1.y - pos2.y;
	if(box_height < 0)box_height *= -1;
    
	//get box orgin pos in the image 
    int2 box_origin_pos;
	if(pos1.x < pos2.x){
		box_origin_pos.x = pos1.x;
	}else{
		box_origin_pos.x = pos2.x;
	}

	if(pos1.y < pos2.y){
		box_origin_pos.y = pos1.y;
	}else{
		box_origin_pos.y = pos2.y;
	}

	if(thickness > 8
	|| thickness < 0){
		thickness = 8;
	}

    int box_x = blockIdx.x * blockDim.x + threadIdx.x;
	int box_y = blockIdx.y * blockDim.y + threadIdx.y;

	int img_x = box_origin_pos.x + box_x;
	int img_y = box_origin_pos.y + box_y;

	if(img_x >= width
	|| img_y >= height){
		return;
	}

	if(box_x >= box_width
	|| box_y >= box_height){
		return;
	}

	float img_x_f = img_x;
	float img_y_f, err_img_y_n_line;
	img_y_f = ((float)pos1.y - pos2.y) / (pos1.x - pos2.x) * (img_x_f - pos1.x) + pos1.y;
	err_img_y_n_line = img_y_f - img_y;
	if(err_img_y_n_line <= 1
    && err_img_y_n_line >= 0 ){
		img[img_y * width + img_x] = pixel_temp;
	}

} 


/************
*
*	application
*   function
*
*************/

/*
description: application to draw the level ruler
para:
*/
void app_draw_level_ruler_on_img(uchar3* img, int width, int height, int ruler_len, int ruler_tooth_height, int thickness, int2 pos)
{
	if(ruler_len > width)ruler_len = width;
	if(ruler_tooth_height > height)ruler_tooth_height = height;

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(ruler_len, blockDim.x), iDivUp(ruler_tooth_height, blockDim.y));

    gpuDrawLevelRuler<<<gridDim, blockDim>>>( img, width, height, ruler_len, ruler_tooth_height, thickness, pos);
}

/*
description: application to draw the vertical ruler
para:
*/
void app_draw_vertical_ruler_on_img(uchar3* img, int width, int height, int ruler_len, int ruler_tooth_height, int thickness, int2 pos)
{
	if(ruler_len > height)ruler_len = height;
	if(ruler_tooth_height > width )ruler_tooth_height = width;

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(ruler_tooth_height, blockDim.x), iDivUp(ruler_len, blockDim.y));

    gpuDrawVerticalRuler<<<gridDim, blockDim>>>( img, width, height, ruler_len, ruler_tooth_height, thickness, pos);
}

/*
description: application to draw a line with two point
para:
*/
void app_draw_a_line_on_img(uchar3* img, int width, int height, int thickness, int2 pos1, int2 pos2)
{

	//get the box width and box height	
	int box_width = pos1.x - pos2.x;
	if(box_width < 0)box_width *= -1;

	int box_height = pos1.y - pos2.y;
	if(box_height < 0)box_height *= -1;

	if(box_height > height)box_height = height;
	if(box_width > width )box_width = width;

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(box_width, blockDim.x), iDivUp(box_height, blockDim.y));

    gpuDrawStraightLines<<<gridDim, blockDim>>>( img, width, height, 0 , pos1, pos2);
}