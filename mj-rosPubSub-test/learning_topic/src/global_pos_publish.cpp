#include <ros/ros.h>
#include <learning_topic/global_pos.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "global_pos_publisher");
    ros::NodeHandle n;
    ros::Publisher global_pos_pub = n.advertise<learning_topic::global_pos>("/globall_balabala", 10);
    ros::Rate loop_rate(1);

    int count = 0;
    while(ros::ok())
    {
        learning_topic::global_pos msg;
        msg.latitude = 112.87;
        msg.longitude = 18.67;
        msg.vel[0] = 1.11;
        msg.vel[1] = 2.22;
        msg.vel[2] = 3.33;
        msg.att[0] = 4.44;
        msg.att[1] = 5.55;
        msg.att[2] = 6.66;

        global_pos_pub.publish(msg);

        ROS_INFO("Publish topic \"globall balabala\": lat: %0.4f, lon: %0.4f, vel[0]: %0.4f,vel[1]: %0.4f, vel[2]: %0.4f, att[0]: %0.4f, att[1]:%0.4f, att[2]:%0.4f",
		                                 msg.latitude, msg.longitude,
		                                   msg.vel[0],msg.vel[1],msg.vel[2], msg.att[0],msg.att[1],msg.att[2]);
        loop_rate.sleep();

    }
    return 0;

}