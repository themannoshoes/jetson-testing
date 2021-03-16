#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

int main(int argc, char ** argv)
{
    //init the node
    ros::init(argc, argv, "velocity_publisher");
    
    //register your node to master
    ros::NodeHandle n;

    ros::Publisher turtle_vel_pub = n.advertise<geometry_msgs::Twist>("/turtle1/cmd_vel",10);

    ros::Rate loop_rate(10);

    int count = 0;
    
    while(ros::ok())
    {
        //create message
        geometry_msgs::Twist vel_msg;
        vel_msg.linear.x = 0.5;
        vel_msg.angular.z = 0.2;
        
        //publish the message
        turtle_vel_pub.publish(vel_msg);
        ROS_INFO("Publish turtle velocity command[%0.2f m/s, %0.2f rad/s]",
                    vel_msg.linear.x, vel_msg.angular.z);

        loop_rate.sleep();

    }
    return 0;


}