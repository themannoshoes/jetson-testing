#include "mj-draw.h"
#include <jetson-utils/cudaFont.h>

extern int blink_state;

imu_info_t         imu_data;
tele_cam_info_t    cam_data;
stream_info_t      stream_data;
g_distance_info_t  g_distance_data;
osd_ctl_info_t     osd_ctl_switch;
std::string        temp_str_c;



int min_of_them(int a, int b){
	return a < b ? a : b;
} 

inline __host__ __device__ float4 alpha_blend( const float4& bg, const float4& fg )
{
	const float alpha = fg.w / 255.0f;
	const float ialph = 1.0f - alpha;
	
	return make_float4(alpha * fg.x + ialph * bg.x,
				    alpha * fg.y + ialph * bg.y,
				    alpha * fg.z + ialph * bg.z,
				    bg.w);
} 

__global__ void gpuBlendBOx( uchar3* input, uchar3* output, int imgWidth, int imgHeight)
{

	float4 color_temp, color_temp1;
	color_temp.x = 0;
	color_temp.y = 0;
	color_temp.z = 200;
	color_temp.w = 0;

	color_temp1.x = 200;
	color_temp1.y = 0;
	color_temp1.z = 0;
	color_temp1.w = 125;

	
	const float px_glyph = 1;
	const float px_glyph1 = 1;

    int box_x = blockIdx.x * blockDim.x + threadIdx.x;
	int box_y = blockIdx.y * blockDim.y + threadIdx.y;

	int img_x = (float)imgWidth/4 + box_x;
	int img_y = (float)imgHeight/4 + box_y;

	if(img_x >= imgWidth
	|| img_y >= imgHeight){
		return;
	}

	if(box_x >= (imgWidth/2)
	|| box_y >= (imgHeight/2)){
		return;
	}


	const float4 px_font = make_float4(px_glyph * color_temp.x, px_glyph * color_temp.y, px_glyph * color_temp.z, px_glyph * color_temp.w);
	const float4 px_in   = cast_vec<float4>(input[img_y * imgWidth + img_x]);

	const float4 px_font1 = make_float4(px_glyph1 * color_temp1.x, px_glyph1 * color_temp1.y, px_glyph1 * color_temp1.z, px_glyph1 * color_temp1.w);
	const float4 px_in1   = cast_vec<float4>(input[img_y * imgWidth + img_x]);
	

	if(box_x > imgWidth/8*3
	&& box_y > imgHeight/8*3){
		output[img_y * imgWidth + img_x] = cast_vec<uchar3>(alpha_blend(px_in1, px_font1));	
	}else{
		output[img_y * imgWidth + img_x] = cast_vec<uchar3>(alpha_blend(px_in, px_font));
	}


}

__global__ void gpuDrawCircle(uchar3 * img, int width, int height, int radius, int thickness)
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

	int img_x = ((float)width - 2*radius)/2 + box_x;
	int img_y = ((float)height - 2*radius)/2 + box_y;

	if(img_x >= width
	|| img_y >= height){
		return;
	}

	if(box_x >= (radius*2)
	|| box_y >= (radius*2)){
		return;
	}

	float result = (box_x - radius) * (box_x - radius);
	result = result + (box_y - radius) * (box_y - radius);
//	result = result - radius * radius;
	
	if( result - (float)radius*radius <= 0
	&&  result - (float)radius*radius >= -200 ){
		img[img_y * width + img_x] = pixel_temp;	
	}

}
__global__ void gpuDrawSolidCircle_pos(uchar3 * img, int width, int height, int radius, int2 center_pos, uchar3 color)
{
	uchar3 pixel_temp;
	pixel_temp = color;

    int box_x = blockIdx.x * blockDim.x + threadIdx.x;
	int box_y = blockIdx.y * blockDim.y + threadIdx.y;

	int img_x = center_pos.x - radius + box_x;
	int img_y = center_pos.y - radius + box_y;

	if(img_x >= width
	|| img_y >= height){
		return;
	}

	if(box_x >= (radius*2)
	|| box_y >= (radius*2)){
		return;
	}

	float result = (box_x - radius) * (box_x - radius);
	result = result + (box_y - radius) * (box_y - radius);
	
	if( result - (float)radius*radius <= 0){
		img[img_y * width + img_x] = pixel_temp;	
	}

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
*	application
*   function
*
*************/

void app_draw_cross_on_img(uchar3* img, int width, int height, int cross_len, int thickness)
{
	if(cross_len > width
	|| cross_len > height){

		cross_len = min_of_them(width, height);
	} 

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(cross_len, blockDim.x), iDivUp(cross_len, blockDim.y));

    gpuDrawCross_mj<<<gridDim, blockDim>>>( img, width, height, cross_len, thickness);
}

void app_draw_cross_on_img_pos(uchar3* img, int width, int height, int cross_len, int thickness, int2 pos)
{
	if(cross_len > width
	|| cross_len > height){

		cross_len = min_of_them(width, height);
	} 

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(cross_len, blockDim.x), iDivUp(cross_len, blockDim.y));

    gpuDrawCross_pos<<<gridDim, blockDim>>>( img, width, height, cross_len, thickness, pos);
}

void app_draw_Box_on_img(uchar3* img, int width, int height, int box_w, int box_h, int thickness)
{
	if(box_w > width
	|| box_h > height){
		return ;
	} 

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(box_w, blockDim.x), iDivUp(box_h, blockDim.y));

    gpuDrawBox<<<gridDim, blockDim>>>( img, width, height, box_w, box_h, thickness);
}

void app_draw_circle_on_img(uchar3 * img, int width, int height, int radius, int thickness)
{

	if((radius * 2) > width
	|| (radius * 2) > height){
		return ;
	} 

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(radius*2, blockDim.x), iDivUp(radius*2, blockDim.y));
	gpuDrawCircle<<<gridDim, blockDim>>>(img, width, height, radius, thickness);

}

void app_blend_on_img(uchar3 * img, int width, int height, int thickness)
{

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(width/2, blockDim.x), iDivUp(height/2, blockDim.y));
 	gpuBlendBOx<<<gridDim, blockDim>>>(img, img, width, height);

}

void app_draw_solidCircle_on_img(uchar3 * img, int width, int height, int radius, int2 center_pos)
{
	uchar3 color_pixel;
	color_pixel.x = 255;
	color_pixel.y = 0;
	color_pixel.z = 0;

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(radius*2, blockDim.x), iDivUp(radius*2, blockDim.y));
 	gpuDrawSolidCircle_pos<<<gridDim, blockDim>>>(img, width, height, radius, center_pos, color_pixel);
}

int app_text_overlay(cudaFont* font, uchar3 * image, int width, int height)
{
	

	temp_str_c = "H.265";

/*    imu_data.year = 2020;
	imu_data.month = 3;
	imu_data.date  = 6;
	imu_data.hour  = 13;
	imu_data.minute = 54;
	imu_data.second = 45;
	imu_data.yaw  = 359.123;
	imu_data.roll = 11.123;
	imu_data.pitch  = 22.123;
	imu_data.longitude = 125.3;
	imu_data.latitude  = 34.7;
	imu_data.height    = 14000;

	cam_data.zoom = 4;
	cam_data.memory_left = 1024;
	cam_data.pics_amount = 8;
	cam_data.pics_num_already_sync = 6;  */

	stream_data.width = 1920;
	stream_data.height = 1080;
	stream_data.frame_rate = 30;
	stream_data.code_type = temp_str_c;
	stream_data.bps = 1;

	char str_temp[256];
	char str_temp1[50];

	//imu_info
	if(osd_ctl_switch.imu_data_osd_switch == 1){
		sprintf(str_temp, "%d-%d-%d %d:%d:%d", imu_data.year, imu_data.month, imu_data.date, imu_data.hour, imu_data.minute, imu_data.second);
		font->OverlayText_edge_alig(image, width, height,
						str_temp, 5, 5, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),3);
		sprintf(str_temp, "yaw: %.3f", imu_data.yaw);
		font->OverlayText_edge_alig(image, width, height,
						str_temp, width, 5, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
		sprintf(str_temp, "pitch: %.3f", imu_data.pitch);
		font->OverlayText_edge_alig(image, width, height,
						str_temp, width, 45, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
		sprintf(str_temp, "roll: %.3f", imu_data.roll);
		font->OverlayText_edge_alig(image, width, height,
						str_temp, width, 85, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
		sprintf(str_temp, "log: %.3f", imu_data.longitude);
		font->OverlayText_edge_alig(image, width, height,
						str_temp, width, 125, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
		sprintf(str_temp, "lat: %.3f", imu_data.latitude);
		font->OverlayText_edge_alig(image, width, height,
						str_temp, width, 165, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
		sprintf(str_temp, "hgt: %.3f", imu_data.height);
		font->OverlayText_edge_alig(image, width, height,
						str_temp, width, 205, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);	
	}
	
	//cam info
	if(osd_ctl_switch.cam_info_osd_switch == 1){
		sprintf(str_temp, "cam zoom: %d", cam_data.zoom);
		font->OverlayText_edge_alig(image, width, height,
						str_temp, 5, 45, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
		sprintf(str_temp, "memory left: %d", cam_data.memory_left);
		font->OverlayText_edge_alig(image, width, height,
						str_temp, 5, 85, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
		sprintf(str_temp, "cam pics num: %d", cam_data.pics_captured_amount);
		font->OverlayText_edge_alig(image, width, height,
						str_temp, 5, 125, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
		sprintf(str_temp, "cam pics sync: %d", cam_data.pics_num_already_sync);
		font->OverlayText_edge_alig(image, width, height,
						str_temp, 5, 165, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
	}

	//stream show
	if(osd_ctl_switch.stream_info_osd_switch == 1){
		for(int i = 0;i < stream_data.code_type.length();i++){
			if(stream_data.code_type.length() > 49) return 0;
			str_temp1[i] = stream_data.code_type[i];
		}
		sprintf(str_temp, "%dx%d@%dfps/%s/%dMbps", stream_data.width, stream_data.height, stream_data.frame_rate, str_temp1, stream_data.bps);
		font->OverlayText_edge_alig(image, width, height,
						str_temp, width, height -30 , make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 50),0);
		float4 temp_rect_pos = font->first_string_pos;
		if(blink_state == 1){
			app_draw_solidCircle_on_img(image, width, height, 10, make_int2(temp_rect_pos.x - 15 -3 ,(temp_rect_pos.y + temp_rect_pos.w)/2) );
		}
	}

	return 0;

}

void init_ros_message_data()
{
	osd_ctl_switch.cam_info_osd_switch = 1;
	osd_ctl_switch.imu_data_osd_switch = 1;
	osd_ctl_switch.stream_info_osd_switch = 1;
	osd_ctl_switch.cross_display_osd_switch = 1;
	osd_ctl_switch.telephoto_cam_view_box_osd_switch = 1;

}
