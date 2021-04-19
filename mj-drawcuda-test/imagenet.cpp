/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>

#include <jetson-utils/cudaFont.h>
#include <jetson-inference/imageNet.h>
#include <jetson-utils/cudaResize.h>

#include <signal.h>

#include "mj-draw.h"

#include <ros/ros.h>
#include <turtlesim/Pose.h>
#include <geometry_msgs/Twist.h>
#include "Person.h"
#include "global_pos.h"
#include "imu_att.h"
#include "tele_cam_zoom.h"
#include "tele_cam_info.h"
#include "utc_time.h"
#include "video_cam.h"

#ifdef HEADLESS
	#define IS_HEADLESS() "headless"	// run without display
#else
	#define IS_HEADLESS() (const char*)NULL
#endif






bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

int usage()
{
	printf("usage: imagenet [--help] [--network=NETWORK] ...\n");
	printf("                input_URI [output_URI]\n\n");
	printf("Classify a video/image stream using an image recognition DNN.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");	
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

	printf("%s", imageNet::Usage());
	printf("%s", videoSource::Usage());
	printf("%s", videoOutput::Usage());
	printf("%s", Log::Usage());

	return 0;
}

pthread_mutex_t timerMutex;
#define TIMERDATA_IT_LOCK() pthread_mutex_lock(&timerMutex)
#define TIMERDATA_IT_UNLOCK() pthread_mutex_unlock(&timerMutex)
int blink_enable = 1;
int blink_state = 1;

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

void poseCallback(const turtlesim:: Pose:: ConstPtr& msg)
{
	static int log_cnt = 0;
	if(log_cnt > 10){
		log_cnt = 0;
		ROS_INFO("Turtle pose: x: %0.6f, y: %0.6f", msg->x, msg->y);
	}
	log_cnt++;
}

void global_pos_Callback(const learning_topic::global_pos::ConstPtr& msg)
{
	static int log_cnt = 0;
	if(log_cnt > 2){
		log_cnt = 0;
		 ROS_INFO("global_pos: lat: %0.4f, lon: %0.4f, vel[0]: %0.4f,vel[1]: %0.4f, vel[2]: %0.4f, att[0]: %0.4f, att[1]:%0.4f, att[2]:%0.4f",
		                                 msg->latitude, msg->longitude,
		                                   msg->vel[0],msg->vel[1],msg->vel[2], msg->att[0],msg->att[1],msg->att[2]);
	}
	log_cnt++;
}



int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv, IS_HEADLESS());

	if( cmdLine.GetFlag("help") )
		return usage();


	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		LogError("can't catch SIGINT\n");


	/*
	 * create input stream
	 */
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

	if( !input )
	{
		LogError("imagenet:  failed to create input stream\n");
		return 0;
	}


	/*
	 * create output stream
	 */
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
		LogError("imagenet:  failed to create output stream\n");	
	

	/*
	 * create font for image overlay
	 */
	cudaFont* font = cudaFont::Create();
	
	if( !font )
	{
		LogError("imagenet:  failed to load font for overlay\n");
		return 0;
	}


	// /*
	//  * create recognition network
	//  */
	// imageNet* net = imageNet::Create(cmdLine);
	
	// if( !net )
	// {
	// 	LogError("imagenet:  failed to initialize imageNet\n");
	// 	return 0;
	// }

	static uint log_flag = 1;
	const int height_restrict = 720;
	const int width_restrict = 1280;


	pthread_t tid;
	pthread_mutex_init(&timerMutex, NULL);
	pthread_create(&tid,NULL,timer_subthread,NULL);


	//init ros node
    int f_argc = 1;
	char **f_argv;
    char *f_cmd[2];
	char f_cmd1[30]= "ABIBA";
    char f_cmd2[30]= "/dev/video1";
    
    f_argv = f_cmd;
    
	f_argv[0] = f_cmd1;
    // f_argv[1] = f_cmd2;
    int i;
    for(i = 0;i< f_argc;i++){
        ROS_INFO("imagenet argc:%d : %s", i, f_argv[i]); 
    }

    ros::init(f_argc, f_argv, "imaghhh");

    ros:: NodeHandle n;

    ros::Subscriber pose_sub = n.subscribe("/globall_balabala", 10, global_pos_Callback);


	/*
	 * processing loop
	 */
	while( !signal_recieved )
	{
		// capture next image image
		uchar3* image = NULL;
		
		if( !input->Capture(&image, 1000) )
		{
			// check for EOS
			if( !input->IsStreaming() )
				break;

			LogError("imagenet:  failed to capture next frame\n");
			continue;
		}
		if(log_flag  == 1){
			log_flag = 0;
			LogError("the img height: %d  img width: %d\n", input->GetHeight(), input->GetWidth());

		}
		cudaResize(image, input->GetWidth(), input->GetHeight(), image, width_restrict, height_restrict);
		// // classify image
		// float confidence = 0.0f;
		// const int img_class = net->Classify(image, input->GetWidth(), input->GetHeight(), &confidence);
	
		// if( img_class >= 0 )
		// {
			// LogVerbose("imagenet:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, img_class, net->GetClassDesc(img_class));	

			// if( font != NULL )
			// {
			// 	char str[256];
			// 	char str1[256];
			// 	sprintf(str1, "string line2 ");
			// 	sprintf(str, "mj no net test");
			// 	font->OverlayText(image, width_restrict, height_restrict,
			// 			        str, 5, 5, make_float4(255, 255, 255, 255), make_float4(0, 0, 0, 100));
			// 	font->OverlayText(image, width_restrict, height_restrict,
			// 			        str1, 50, 50, make_float4(0, 255, 0, 100), make_float4(0, 0, 0, 0),0);
				
			// }
		// }
		mj_text_app(font,image, width_restrict, height_restrict);

     	// mj_drawBlend_test(image, width_restrict, height_restrict, 3);
		mj_drawCircle_test(image, width_restrict, height_restrict, 100, 3);
		mj_drawBox_test(image, width_restrict, height_restrict, width_restrict/ 2,height_restrict /2, 3);
		mj_draw_test(image, width_restrict, height_restrict, height_restrict /4, 4);

		// render outputs
		if( output != NULL )
		{
			output->Render(image, width_restrict, height_restrict);

			// update status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | 000 | Network 000 FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
			output->SetStatus(str);	

			// check if the user quit
			if( !output->IsStreaming() )
				signal_recieved = true;
		}

		// // print out timing info
		// net->PrintProfilerTimes();
		
		ros::spinOnce();
	}
	
	
	/*
	 * destroy resources
	 */
	LogVerbose("imagenet:  shutting down...\n");
	int err_thread;
	pthread_cancel(tid);
	err_thread =  pthread_join(tid,NULL);
	if(err_thread){
		LogError("imagenet:  cannot thread join\n");
	}

	LogVerbose("imagenet:  freeing memory.\n");
	SAFE_DELETE(input);
	SAFE_DELETE(output);
	SAFE_DELETE(font);
	// SAFE_DELETE(net);
	
	LogVerbose("imagenet:  shutdown complete.\n");
	return 0;
}

float4 cudaFont::first_string_pos;
int mj_text_app(cudaFont* font, uchar3 * image, int width, int height)
{
	
	imu_info_t       imu_data;
	tele_cam_info_t  cam_data;
	stream_info_t    stream_data;
	g_distance_info_t g_distance_data;
	std::string temp_str_c;
	temp_str_c = "H.265";

    imu_data.year = 2020;
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
	cam_data.pics_num_already_sync = 6;

	stream_data.width = 1920;
	stream_data.height = 1080;
	stream_data.frame_rate = 30;
	stream_data.code_type = temp_str_c;
	stream_data.bps = 1;

	char str_temp[256];
	char str_temp1[50];

	//imu_info
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
	
	//cam info
	sprintf(str_temp, "cam zoom: %d", cam_data.zoom);
	font->OverlayText_edge_alig(image, width, height,
					str_temp, 5, 45, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
	sprintf(str_temp, "memory left: %d", cam_data.memory_left);
	font->OverlayText_edge_alig(image, width, height,
					str_temp, 5, 85, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
	sprintf(str_temp, "cam pics num: %d", cam_data.pics_amount);
	font->OverlayText_edge_alig(image, width, height,
					str_temp, 5, 125, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
	sprintf(str_temp, "cam pics sync: %d", cam_data.pics_num_already_sync);
	font->OverlayText_edge_alig(image, width, height,
					str_temp, 5, 165, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);

	//stream show
	for(int i = 0;i < stream_data.code_type.length();i++){
		if(stream_data.code_type.length() > 49) return 0;
		str_temp1[i] = stream_data.code_type[i];
	}
	sprintf(str_temp, "%dx%d@%dfps/%s/%dMbps", stream_data.width, stream_data.height, stream_data.frame_rate, str_temp1, stream_data.bps);
	font->OverlayText_edge_alig(image, width, height,
					str_temp, width, height -30 , make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 50),0);
	float4 temp_rect_pos = font->first_string_pos;
	if(blink_state == 1){
		mj_draw_SolidCircle_test(image, width, height, 10, make_int2(temp_rect_pos.x - 15 -3 ,(temp_rect_pos.y + temp_rect_pos.w)/2) );
	}

	return 0;

}
