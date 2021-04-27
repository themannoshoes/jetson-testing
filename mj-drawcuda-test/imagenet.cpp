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
#include "osd_ros_process.h"

#include <ros/ros.h>
#include <turtlesim/Pose.h>
#include <geometry_msgs/Twist.h>
#include "ros_msg_include/global_pos.h"
#include "ros_msg_include/imu_att.h"
#include "ros_msg_include/tele_cam_zoom.h"
#include "ros_msg_include/tele_cam_info.h"
#include "ros_msg_include/utc_time.h"
#include "ros_msg_include/video_cam.h"
#include "ros_msg_include/osd_cmd.h"


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
	char f_cmd1[30]= "fake_arg1";
    char f_cmd2[30]= "fake_arg2";
    
    f_argv = f_cmd;
    
	f_argv[0] = f_cmd1;
    // f_argv[1] = f_cmd2;
    int i;
    for(i = 0;i< f_argc;i++){
        ROS_INFO("imagenet argc:%d : %s", i, f_argv[i]); 
    }

    ros::init(f_argc, f_argv, "video_viewer_osd");

    ros:: NodeHandle n;

	ros::Subscriber global_pos_sub = n.subscribe("/gimbal/global_pos", 10, ros_global_pos_Callback);
	ros::Subscriber imu_att_sub = n.subscribe("/gimbal/imu_att", 10, ros_imu_att_Callback);
	ros::Subscriber osd_cmd_sub = n.subscribe("/gimbal/cmd_osd", 10, ros_osd_cmd_Callback);
	ros::Subscriber tele_cam_info_sub = n.subscribe("/gimbal/tele_cam_info", 10, ros_tele_cam_info_Callback);
	ros::Subscriber tele_cam_zoom_sub = n.subscribe("/gimbal/tele_cam_zoom", 10, ros_tele_cam_zoom_Callback);
	ros::Subscriber utc_time_sub = n.subscribe("/gimbal/utc_time", 10, ros_utc_time_Callback);
	ros::Subscriber video_cam_sub = n.subscribe("/gimbal/video_cam", 10, ros_video_cam_Callback);

	init_ros_message_data();
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

		app_text_overlay(font,image, width_restrict, height_restrict);

     	// app_blend_on_img(image, width_restrict, height_restrict, 3);
		// app_draw_circle_on_img(image, width_restrict, height_restrict, 100, 3);
		if(osd_ctl_switch.telephoto_cam_view_box_osd_switch == 1){
			app_draw_Box_on_img(image, width_restrict, height_restrict, width_restrict/ 2,height_restrict /2, 3);
		}
		if(osd_ctl_switch.cross_display_osd_switch == 1){
			app_draw_cross_on_img(image, width_restrict, height_restrict, height_restrict /4, 4);
		}

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



