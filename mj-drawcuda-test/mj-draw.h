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

	int zoom;
	int memory_left;
	int pics_captured_amount;
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

typedef struct _osd_ctl_cmd_info_t{
	uint8_t cam_info_osd_switch;
	uint8_t imu_data_osd_switch;
	uint8_t stream_info_osd_switch;
	uint8_t cross_display_osd_switch;
	uint8_t telephoto_cam_view_box_osd_switch;

}osd_ctl_info_t;

extern imu_info_t         imu_data;
extern tele_cam_info_t    cam_data;
extern stream_info_t      stream_data;
extern g_distance_info_t  g_distance_data;
extern osd_ctl_info_t     osd_ctl_switch;
extern std::string        temp_str_c;

void app_draw_cross_on_img(uchar3* img, int width, int height, int cross_len,int thickness);
void app_draw_cross_on_img_pos(uchar3* img, int width, int height, int cross_len, int thickness, int2 pos);
void app_draw_Box_on_img(uchar3* img, int width, int height, int box_w, int box_h, int thickness);
void app_draw_circle_on_img(uchar3 * img, int width, int height, int radius, int thickness);
void app_blend_on_img(uchar3 * img, int width, int height, int thickness);
void app_draw_solidCircle_on_img(uchar3 * img, int width, int height, int radius, int2 center_pos);

void app_draw_level_ruler_on_img(uchar3* img, int width, int height, int ruler_len, int ruler_tooth_height, int thickness, int2 pos);
void app_draw_vertical_ruler_on_img(uchar3* img, int width, int height, int ruler_len, int ruler_tooth_height, int thickness, int2 pos);
void app_draw_a_line_on_img(uchar3* img, int width, int height, int thickness, int2 pos1, int2 pos2);
void app_draw_a_triangle_on_img(uchar3* img, int width, int height, int thickness, int2 pos1, int2 pos2, int2 pos3);
void draw_triangle_in_a_box(uchar3* img, int width, int height, int thickness, int orientate_flag, int2 pos1, int2 pos2);

int app_text_overlay(cudaFont* font, uchar3 * image, int width, int height);

void init_ros_message_data(void);
#endif