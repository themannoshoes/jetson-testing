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
	const int height_restrict = 1080;
	const int width_restrict = 1920;
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
		mj_text_app(image, width_restrict, height_restrict);

     	mj_drawBlend_test(image, width_restrict, height_restrict, 3);
		mj_drawCircle_test(image, width_restrict, height_restrict, 100, 3);
		mj_drawBox_test(image, width_restrict, height_restrict, width_restrict/ 2,height_restrict /2, 3);
		mj_draw_test(image, width_restrict, height_restrict, height_restrict /4, 4);
		//CUDA(cudaDeviceSynchronize());
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
	}
	
	
	/*
	 * destroy resources
	 */
	LogVerbose("imagenet:  shutting down...\n");
	
	SAFE_DELETE(input);
	SAFE_DELETE(output);
	// SAFE_DELETE(net);
	
	LogVerbose("imagenet:  shutdown complete.\n");
	return 0;
}

int mj_text_app(uchar3 * image, int width, int height)
{
	
	imu_info_t       imu_data;
	tele_cam_info_t  cam_data;
	stream_info_t    stream_data;
	g_distance_info_t g_distance_data;

	cudaFont* font = cudaFont::Create();
	if( !font )
	{
		LogError("imagenet:  failed to load font for overlay\n");
		return 0;
	}

    imu_data.year = 2020;
	imu_data.month = 3;
	imu_data.date  = 6;
	imu_data.hour  = 13;
	imu_data.minute = 54;
	imu_data.second = 45;
	imu_data.yaw  = 1234.2344;
	imu_data.roll = 987654321.4556;
	imu_data.pitch  = 123459876.7648;
	imu_data.longitude = 125.3;
	imu_data.latitude  = 34.7;
	imu_data.height    = 14000;

	char str_temp[256];
	
	std::string str_time;
	sprintf(str_temp, "%d-%d-%d %d:%d:%d", imu_data.year, imu_data.month, imu_data.date, imu_data.hour, imu_data.minute, imu_data.second);
	font->OverlayText(image, width, height,
					str_temp, width - 300, 40, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
	sprintf(str_temp, "yaw: %.3f", imu_data.yaw);
	font->OverlayText(image, width, height,
					str_temp, 5, 80, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
	sprintf(str_temp, "pitch: %.3f", imu_data.pitch);
	font->OverlayText(image, width, height,
					str_temp, 5, 120, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
	sprintf(str_temp, "roll: %.3f", imu_data.pitch);
	font->OverlayText(image, width, height,
					str_temp, 5, 160, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
	sprintf(str_temp, "log: %.3f", imu_data.longitude);
	font->OverlayText(image, width, height,
					str_temp, 5, 200, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
	sprintf(str_temp, "lat: %.3f", imu_data.latitude);
	font->OverlayText(image, width, height,
					str_temp, 5, 240, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);
	sprintf(str_temp, "hgt: %.3f", imu_data.height);
	font->OverlayText(image, width, height,
					str_temp, 5, 280, make_float4(0, 255, 0, 255), make_float4(0, 0, 0, 10),0);										

	str_time = str_temp;

	if( font != NULL )
	{
		char str[256];
		sprintf(str, "mj no net test");
		font->OverlayText(image, width, height,
						str, 5, 5, make_float4(255, 255, 255, 255), make_float4(0, 0, 0, 100));

	}
	return 0;

}
