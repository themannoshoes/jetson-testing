#ifndef _OSD_ROS_PROCESS_H_
#define _OSD_ROS_PROCESS_H_

#include <ros/ros.h>
#include "ros_msg_include/global_pos.h"
#include "ros_msg_include/imu_att.h"
#include "ros_msg_include/tele_cam_zoom.h"
#include "ros_msg_include/tele_cam_info.h"
#include "ros_msg_include/utc_time.h"
#include "ros_msg_include/video_cam.h"
#include "ros_msg_include/osd_cmd.h"
#include <stdio.h>
#include <string>
#include <math.h>

#define TIMERDATA_IT_LOCK() pthread_mutex_lock(&timerMutex)
#define TIMERDATA_IT_UNLOCK() pthread_mutex_unlock(&timerMutex)




extern pthread_mutex_t timerMutex;
extern int blink_enable;
extern int blink_state;

void *timer_subthread(void * arg);
void ros_global_pos_Callback(const learning_topic::global_pos::ConstPtr& msg);
void ros_imu_att_Callback(const learning_topic::imu_att::ConstPtr& msg);
void ros_osd_cmd_Callback(const learning_topic::osd_cmd::ConstPtr& msg);
void ros_tele_cam_info_Callback(const learning_topic::tele_cam_info::ConstPtr& msg);
void ros_tele_cam_zoom_Callback(const learning_topic::tele_cam_zoom::ConstPtr& msg);
void ros_utc_time_Callback(const learning_topic::utc_time::ConstPtr& msg);
void ros_video_cam_Callback(const learning_topic::video_cam::ConstPtr& msg);


#endif