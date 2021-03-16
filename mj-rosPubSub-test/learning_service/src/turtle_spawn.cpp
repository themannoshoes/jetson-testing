#include <ros/ros.h>
#include <turtlesim/Spawn.h>

int main(int argc, char ** argv)
{
    //we init a node
    ros::init(argc, argv, "turtle_spawn");

    //we new a nodehandle
    ros::NodeHandle node;

    //we try to find a service named "/spawn", when "/spawn" is found, we new a client to connect the server
    ros::service::waitForService("/spawn");
    ros::ServiceClient add_turtle = node.serviceClient<turtlesim::Spawn>("/spawn");

    //we new a srv request data structure, and init
    turtlesim::Spawn srv;
    srv.request.x = 2.0;
    srv.request.y = 2.0;
    srv.request.name = "turtle2";

    ROS_INFO("Call service to spawn turtle[x: %0.6f , y: %0.6f, name: %s]",
                srv.request.x, srv.request.y, srv.request.name.c_str());

    //we send "srv" to do the request and then wait
    add_turtle.call(srv);
    
    //display the reponse from server
    ROS_INFO("Spawn turtle successfully [name: %s]", srv.response.name.c_str());

    return 0;


}