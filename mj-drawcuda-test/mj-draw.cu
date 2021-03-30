#include "mj-draw.h"



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

void mj_drawCircle_test(uchar3 * img, int width, int height, int radius, int thickness)
{

	if((radius * 2) > width
	|| (radius * 2) > height){
		return ;
	} 

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(radius*2, blockDim.x), iDivUp(radius*2, blockDim.y));
	gpuDrawCircle<<<gridDim, blockDim>>>(img, width, height, radius, thickness);

}

void mj_drawBlend_test(uchar3 * img, int width, int height, int thickness)
{

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(width/2, blockDim.x), iDivUp(height/2, blockDim.y));
 	gpuBlendBOx<<<gridDim, blockDim>>>(img, img, width, height);

}

void mj_draw_SolidCircle_test(uchar3 * img, int width, int height, int radius, int2 center_pos)
{
	uchar3 color_pixel;
	color_pixel.x = 255;
	color_pixel.y = 0;
	color_pixel.z = 0;

	dim3 blockDim(8,8);
	dim3 gridDim(iDivUp(radius*2, blockDim.x), iDivUp(radius*2, blockDim.y));
 	gpuDrawSolidCircle_pos<<<gridDim, blockDim>>>(img, width, height, radius, center_pos, color_pixel);
}
