#ifndef __MJ_DRAW_H__
#define __MJ_DRAW_H__


#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/imageFormat.h>

void mj_draw_test(uchar3* img, int width, int height, int cross_len,int thickness);

#endif