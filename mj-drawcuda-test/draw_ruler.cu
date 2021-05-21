#include "mj-draw.h"
#include <jetson-utils/cudaFont.h>
#include <float.h>
#include <math.h>

#define YAW_MIN_VAL 0.0f
#define YAW_MAX_VAL 360.0f
#define PITCH_MIN_VAL -90.0f
#define PITCH_MAX_VAL  90.0f


/*
description: find the min and max of 3 value
*/
int2 find_the_min_n_max_of_3_value(int value1, int value2 , int value3)
{
	int min_x = 0;
	int max_x = 0;
	int2 ret_min_and_max;

	if(value1 > value2){
		min_x = value2;
		max_x = value1;
	}else{
		min_x = value1;
		max_x = value2;
	}

	if(min_x > value3){
		min_x = value3;
	}

	if(max_x < value3){
		max_x = value3;
	}
	ret_min_and_max = make_int2(min_x, max_x);
	return ret_min_and_max;
}

void angle_restrain_in_0_to_360(float & angle)
{
	if(angle < 0.0f)angle += 360.0f;
	if(angle > 360.0f)angle -= 360.0f;  
}





/*
* draw a graduated ruler, the orgin position is parameter "int2 pos"
*/
__global__ void gpuDrawLevelRuler (uchar3 * img, int width, int height, int ruler_len, int ruler_tooth_height, int thickness, int2 pos)
{
	uchar3 pixel_temp;
	pixel_temp.x = 0;
	pixel_temp.y = 250;
	pixel_temp.z = 0;
    
	if(thickness > (ruler_tooth_height /4)    //restrain the thickness
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
	int i, img_offset;

	if(box_y == (ruler_tooth_height-1 - thickness /2) ){
		for(i = 0; i < thickness;i++){
			img_offset = img_y + i - thickness/2;
			if(img_offset >= height || img_offset <0)break;
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
			img_offset = img_x + i - thickness/2;
			if(img_offset >= width || img_offset < 0)break;
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
	int i, img_offset;

	if(box_x == (ruler_tooth_height-1 - thickness/2)){
		for(i = 0;i < thickness;i++){
			img_offset = img_x + i - thickness/2;
			if(img_offset >= width || img_offset < 0)break;
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
			img_offset = img_y + i - thickness/2;
			if(img_offset >= height || img_offset < 0)break;
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
	box_width = box_width + 1;

	int box_height = pos1.y - pos2.y;
	if(box_height < 0)box_height *= -1;
	box_height = box_height + 1;
    
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

	float img_x_f,img_y_f, err_img_y_n_line, err_img_x_n_line;
	
	if(pos1.x != pos2.x){
		img_x_f = img_x;	
		img_y_f = ((float)pos1.y - pos2.y) / (pos1.x - pos2.x) * (img_x_f - pos1.x) + pos1.y;
		err_img_y_n_line = img_y_f - img_y;
		if(err_img_y_n_line < 1
		&& err_img_y_n_line >=0 ){
			img[img_y * width + img_x] = pixel_temp;
		}
	}else{
		if(img_x == pos1.x){
			img[img_y * width + img_x] = pixel_temp;
		}
	}

	if(pos1.y != pos2.y){
		img_y_f = img_y;
		img_x_f = ((float)pos1.x - pos2.x) / (pos1.y - pos2.y) * (img_y_f - pos1.y) + pos1.x;
		err_img_x_n_line = img_x_f - img_x;
		if(err_img_x_n_line < 1
		&& err_img_x_n_line >=0){
			img[img_y * width + img_x] = pixel_temp;
		}
	}else{
		if(img_y == pos1.y){
			img[img_y * width + img_x] = pixel_temp;
		}
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
	box_width = box_width + 1;  //prevent the case that "pos1.x == pos2.x"

	int box_height = pos1.y - pos2.y;
	if(box_height < 0)box_height *= -1;
	box_height = box_height + 1;

	if(box_height > height)box_height = height;
	if(box_width > width )box_width = width;

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(box_width, blockDim.x), iDivUp(box_height, blockDim.y));

    gpuDrawStraightLines<<<gridDim, blockDim>>>( img, width, height, 0 , pos1, pos2);
}



/*
description: application to draw a triangle with 3 point
para:
*/
void app_draw_a_triangle_on_img(uchar3* img, int width, int height, int thickness, int2 pos1, int2 pos2, int2 pos3)
{
	int2 min_n_max_x;
	int2 min_n_max_y;
	int min_x, max_x, min_y, max_y;
	min_n_max_x = find_the_min_n_max_of_3_value(pos1.x, pos2.x, pos3.x);
	min_n_max_y = find_the_min_n_max_of_3_value(pos1.y, pos2.y, pos3.y);
	min_x = min_n_max_x.x;
	max_x = min_n_max_x.y;
	min_y = min_n_max_y.x;
	max_y = min_n_max_y.y;

	//get the box width and box height
	int box_width = max_x - min_x + 1;
	int box_height = max_y - min_y + 1;

	if(box_height > height)box_height = height;
	if(box_width > width )box_width = width;

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(box_width, blockDim.x), iDivUp(box_height, blockDim.y));

    gpuDrawStraightLines<<<gridDim, blockDim>>>( img, width, height, 0 , pos1, pos2);
	gpuDrawStraightLines<<<gridDim, blockDim>>>( img, width, height, 0 , pos2, pos3);
	gpuDrawStraightLines<<<gridDim, blockDim>>>( img, width, height, 0 , pos3, pos1);
}



/******************
	application
	function

	level:2
******************/

/*
description: draw a triangle in a box based on 2 point
para "orientate_flag" :  0x01:level; 0x00: vertical
*/
void app_draw_triangle_in_a_box(uchar3* img, int width, int height, int thickness, int orientate_flag, int2 pos1, int2 pos2)
{
	int2 triag_pos1, triag_pos2, triag_pos3, temp_int2;
	int box_min_x, box_min_y,box_max_x, box_max_y;

	if(pos1.x == pos2.x
	|| pos1.y == pos2.y){
		return;
	}
	temp_int2 = find_the_min_n_max_of_3_value(pos1.x, pos2.x, pos2.x);
	box_min_x = temp_int2.x;
	box_max_x = temp_int2.y;
	temp_int2 = find_the_min_n_max_of_3_value(pos1.y, pos2.y, pos2.y);
	box_min_y = temp_int2.x;
	box_max_y = temp_int2.y; 

	if(orientate_flag == 1){  //this case is "level"
		triag_pos1.x = box_min_x;
		triag_pos1.y = box_min_y;

		triag_pos2.x = box_min_x;
		triag_pos2.y = box_max_y;

		triag_pos3.x = box_max_x;
		triag_pos3.y = (box_max_y + box_min_y)/2;
	}else{  //this case is "vertical"
		triag_pos1.x = box_min_x;
		triag_pos1.y = box_min_y;

		triag_pos2.x = box_max_x;
		triag_pos2.y = box_min_y;
		
		triag_pos3.x = (box_max_x + box_min_x)/2;
		triag_pos3.y = box_max_y;
	}
	app_draw_a_triangle_on_img(img, width, height, thickness, triag_pos1, triag_pos2, triag_pos3);

}

/*****************
	application
	function
	level:3
*****************/

/*
description: draw a indicating graduated ruler for pitch & yaw data
para:
*/
void app_draw_indicating_ruler_for_pitch_N_yaw(uchar3* img, int width, int height, int thickness)
{
	//imu_data.yaw  = 320;
	//imu_data.pitch  = 22.123;

	const int lv_ruler_tooth_height = 6;
	const int vtc_ruler_tooth_height = 6;
	const int triag_box_bottom_len = 5;
	const int triag_box_side_len = 10;

	/* input parameter */
	float yaw_catched   = imu_data.yaw;
	float pitch_catched = imu_data.pitch;
	int lv_ruler_len    = width /2;
	int vtc_ruler_len   = height/2;
	int2 lv_ruler_pos = make_int2(10, height - lv_ruler_tooth_height - 3);
	int2 vtc_ruler_pos = make_int2(width - vtc_ruler_tooth_height - 3, 10);
	

	int yaw_pixel_len, pitch_pixel_len;
	int2 yaw_indic_triag_box_pos1, yaw_indic_triag_box_pos2, pitch_indic_triag_box_pos1, pitch_indic_triag_box_pos2;

	angle_restrain_in_0_to_360(yaw_catched);
	yaw_pixel_len   = (yaw_catched / YAW_MAX_VAL) * lv_ruler_len;
	pitch_pixel_len = ((pitch_catched + 90) / (PITCH_MAX_VAL - PITCH_MIN_VAL)) * vtc_ruler_len;
	
	yaw_indic_triag_box_pos1.x   = lv_ruler_pos.x + yaw_pixel_len - triag_box_bottom_len /2;
	yaw_indic_triag_box_pos1.y   = lv_ruler_pos.y - triag_box_side_len - 1;  //leave 1 pixel between ruler & triangle 
	yaw_indic_triag_box_pos2.x = yaw_indic_triag_box_pos1.x + triag_box_bottom_len -1;
	yaw_indic_triag_box_pos2.y = yaw_indic_triag_box_pos1.y + triag_box_side_len -1;

	pitch_indic_triag_box_pos1.x = vtc_ruler_pos.x - triag_box_side_len - 1;
	pitch_indic_triag_box_pos1.y = vtc_ruler_pos.y + pitch_pixel_len - triag_box_bottom_len /2;
	pitch_indic_triag_box_pos2.x = pitch_indic_triag_box_pos1.x + triag_box_side_len -1;
	pitch_indic_triag_box_pos2.y = pitch_indic_triag_box_pos1.y + triag_box_bottom_len -1;

	app_draw_level_ruler_on_img(img, width, height, lv_ruler_len, lv_ruler_tooth_height, thickness, lv_ruler_pos);
	app_draw_vertical_ruler_on_img(img, width, height, vtc_ruler_len, vtc_ruler_tooth_height, thickness, vtc_ruler_pos);
	app_draw_triangle_in_a_box(img, width, height, thickness, 0x00, yaw_indic_triag_box_pos1, yaw_indic_triag_box_pos2);
	app_draw_triangle_in_a_box(img, width, height, thickness, 0x01, pitch_indic_triag_box_pos1, pitch_indic_triag_box_pos2);
}

