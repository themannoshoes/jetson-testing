#include "osd_ros_process.h"
#include "mj-draw.h"

pthread_mutex_t timerMutex;

int blink_enable = 1;
int blink_state = 1;
float4 cudaFont::first_string_pos;

void set_Timer(int seconds, int mseconds)
{
	struct timeval temp;

	temp.tv_sec = seconds;
	temp.tv_usec = mseconds * 1000;

	select(0, NULL, NULL, NULL, &temp);
	return;
}

//timer thread
void *timer_subthread(void * arg)
{
	int count = 0;
	int i = 0;
	while(1){
		set_Timer(0,500);
		count++;
		TIMERDATA_IT_LOCK();
		if(blink_enable == 1){
			blink_state = !blink_state;
		}else{
			blink_state = 1;
		}
		TIMERDATA_IT_UNLOCK();
	}
}


void ros_global_pos_Callback(const learning_topic::global_pos::ConstPtr& msg)
{
	// static int log_cnt = 0;
	// if(log_cnt > 2){
	// 	log_cnt = 0;
	// 	 ROS_INFO("global_pos: lat: %0.4f, lon: %0.4f, vel[0]: %0.4f,vel[1]: %0.4f, vel[2]: %0.4f, att[0]: %0.4f, att[1]:%0.4f, att[2]:%0.4f",
	// 	                                 msg->latitude, msg->longitude,
	// 	                                   msg->vel[0],msg->vel[1],msg->vel[2], msg->att[0],msg->att[1],msg->att[2]);
	// }
	// log_cnt++;
	imu_data.longitude = msg->longitude;
	imu_data.latitude = msg->latitude;
}

void ros_imu_att_Callback(const learning_topic::imu_att::ConstPtr& msg)
{
	// static int log_cnt = 0;
	// if(log_cnt > 2){
	// 	log_cnt = 0;
	// 	 ROS_INFO("imu_att: roll: %0.4f, pitch: %0.4f, yaw: %0.4f",
	// 	                                 msg->roll, msg->pitch, msg->yaw);
	// }
	// log_cnt++;
	imu_data.roll = msg->roll;
	imu_data.pitch = msg->pitch;
	imu_data.yaw = msg->yaw;
}

void ros_osd_cmd_Callback(const learning_topic::osd_cmd::ConstPtr& msg)
{
	// static int log_cnt = 0;
	// if(log_cnt > 2){
	// 	log_cnt = 0;
	// 	for(int i = 0;i < msg->data.size(); i++){
	// 			 ROS_INFO("osd_att: data[%d]: %d \r\n", i, msg->data[i]);	
	// 	}
	// }
	// log_cnt++;
	for(int i = 0;i < msg->data.size(); i++){
		switch(i){
		case 0:
			osd_ctl_switch.cam_info_osd_switch = msg->data[0];
			break;
		case 1:
		 	osd_ctl_switch.imu_data_osd_switch = msg->data[1];
			break;
		case 2:
			osd_ctl_switch.stream_info_osd_switch = msg->data[2];
			break;
		case 3:
			osd_ctl_switch.cross_display_osd_switch = msg->data[3];
			break;
		case 4:
			osd_ctl_switch.telephoto_cam_view_box_osd_switch = msg->data[4];
			break;
		}	
	}
}
void ros_tele_cam_info_Callback(const learning_topic::tele_cam_info::ConstPtr& msg)
{
	// static int log_cnt = 0;
	// if(log_cnt > 2){
	// 	log_cnt = 0;
	// 	 ROS_INFO("tele_cam_info: space: %d, photo_captured: %d, photo_synced: %d",
	// 	                                 msg->space, msg->photo_captured, msg->photo_synced);
	// }
	// log_cnt++;
	cam_data.memory_left = msg->space;
	cam_data.pics_captured_amount = msg->photo_captured;
	cam_data.pics_num_already_sync = msg->photo_synced;
}
void ros_tele_cam_zoom_Callback(const learning_topic::tele_cam_zoom::ConstPtr& msg)
{
	// static int log_cnt = 0;
	// if(log_cnt > 2){
	// 	log_cnt = 0;
	// 	 ROS_INFO("tele_cam_zoom: zoom_pos: %d", msg->zoom_pos);	
	// }
	// log_cnt++;
	cam_data.zoom = msg->zoom_pos;
}
void ros_utc_time_Callback(const learning_topic::utc_time::ConstPtr& msg)
{
	// static int log_cnt = 0;
	// if(log_cnt > 2){
	// 	log_cnt = 0;
	// 	 ROS_INFO("utc_time: year: %d, month: %d, day: %d, hour: %d, min: %d, sec: %d",
	// 	                                msg->year, msg->month,
	// 	                        		msg->day, msg->hour, msg->min, msg->sec);
	// }
	// log_cnt++;
	imu_data.year  = msg->year;
	imu_data.month = msg->month;
	imu_data.date = msg->day;
	imu_data.hour = msg->hour;
	imu_data.minute = msg->min;
	imu_data.second = msg->sec;
}
void ros_video_cam_Callback(const learning_topic::video_cam::ConstPtr& msg)
{
	// static int log_cnt = 0;
	// if(log_cnt > 2){
	// 	log_cnt = 0;
	// 	 ROS_INFO("video_cam: zoom_pos: %d, tempature: %d, stabilize: %d, focus_mode: %d, defog: %d",
	// 	                                msg->zoom_pos, msg->tempature,
	// 	                                msg->stabilize, msg->focus_mode, msg->defog);
	// }
	// log_cnt++;
	
}