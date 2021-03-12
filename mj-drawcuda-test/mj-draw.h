#ifndef __MJ_DRAW_H__
#define __MJ_DRAW_H__


#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/imageFormat.h>
#include <jetson-utils/cudaMath.h>

void mj_draw_test(uchar3* img, int width, int height, int cross_len,int thickness);
void mj_draw_test_pos(uchar3* img, int width, int height, int cross_len, int thickness, int2 pos);
#endif