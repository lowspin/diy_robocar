/* Adapted from https://github.com/wilselby/diy_driverless_car_ROS/blob/master/rover_teleop/src/rover_joy_teleop.cpp */

#include<ros/ros.h>
#include<geometry_msgs/Twist.h>
#include<geometry_msgs/TwistStamped.h>
#include<sensor_msgs/Joy.h>
#include<iostream>

using namespace std;
float max_linear_vel = 0.2;
float max_angular_vel = 1.5707;


ros::Publisher pub;
ros::Publisher pubStamped;
ros::Subscriber sub;
int i_velLinear;
int i_velAngular;

void callBack(const sensor_msgs::JoyConstPtr& joy)
{
	  
  geometry_msgs::Twist vel;
  vel.angular.z = -max_angular_vel*joy->axes[0];
  vel.linear.x = max_linear_vel*joy->axes[4];
  pub.publish(vel);

  geometry_msgs::TwistStamped velStamped;
  ros::Time current_time = ros::Time::now();

  velStamped.header.stamp.sec = current_time.sec;
  velStamped.header.stamp.nsec = current_time.nsec;

  velStamped.twist = vel;
  pubStamped.publish(velStamped);

}

int main(int argc, char** argv) {

  ros::init(argc, argv, "rover_joy_teleop");	//Specify node name
  
  ros::NodeHandle n;
  
  i_velLinear = 1;
  i_velAngular = 0;
  n.param("axis_linear", i_velLinear, i_velLinear);
  n.param("axis_angular", i_velAngular, i_velAngular);
  sub = n.subscribe("joy", 10, callBack);
  pub = n.advertise<geometry_msgs::Twist>("cmd_vel",1);
  pubStamped = n.advertise<geometry_msgs::TwistStamped>("cmd_vel_stamped",1);

  ros::spin();

  return 0;
}

