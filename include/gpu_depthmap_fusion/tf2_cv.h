#pragma once

#include <opencv2/opencv.hpp>
#include <tf2/convert.h>
#include <tf2/LinearMath/Transform.h>
#include <geometry_msgs/Transform.h>

namespace tf2
{

inline
void transformToCv(const geometry_msgs::Transform& in, cv::Matx44f& out)
{
    tf2::Transform tf;
    convert(in, tf);
    const auto& origin = tf.getOrigin();
    const auto& rotation = tf.getBasis();
    out(0,3) = origin.getX();
    out(1,3) = origin.getY();
    out(2,3) = origin.getZ();
    out(3,3) = 1;
    for(int i=0; i<3; ++i)
    {
        out(i,0) = rotation[i].getX();
        out(i,1) = rotation[i].getY();
        out(i,2) = rotation[i].getZ();
    }
}


}