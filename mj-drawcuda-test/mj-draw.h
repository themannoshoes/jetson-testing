#ifndef __MJ_DRAW_H__
#define __MJ_DRAW_H__


#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/imageFormat.h>
#include <jetson-utils/cudaMath.h>
#include <jetson-utils/cudaFont.h>
#include <jetson-utils/cudaVector.h>
#include <stdio.h>
#include <string>
#include <math.h>

typedef struct _imu_base_info_t{
	int year;
	int month;
	int date;
	int hour;
	int minute;
	int second;

	float yaw;
	float pitch;
	float roll;
	float longitude;
	float latitude;
	float height;
}imu_info_t;

typedef struct _tele_cam_base_info_t{

	int zomm;
	int memory_left;
	int pics_amount;
	int pics_num_already_sync;
}tele_cam_info_t;

typedef struct _v_stream_base_info_t{
	int width;
	int height;
	int frame_rate;
	std::string code_type;
	int bps;
}stream_info_t;

typedef struct _ground_distance_info_t{
	float longitude;
	float latitude;
	float distance;
}g_distance_info_t;



void mj_draw_test(uchar3* img, int width, int height, int cross_len,int thickness);
void mj_draw_test_pos(uchar3* img, int width, int height, int cross_len, int thickness, int2 pos);
void mj_drawBox_test(uchar3* img, int width, int height, int box_w, int box_h, int thickness);
void mj_drawCircle_test(uchar3 * img, int width, int height, int radius, int thickness);
void mj_drawBlend_test(uchar3 * img, int width, int height, int thickness);

int mj_text_app(uchar3 * image, int width, int height);
#endif