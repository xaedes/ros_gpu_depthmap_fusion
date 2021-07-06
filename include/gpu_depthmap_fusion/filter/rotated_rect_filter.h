#pragma once

#include <opencv2/opencv.hpp>

#include "gpu_depthmap_fusion/filter/filter.h"
#include "gpu_depthmap_fusion/filter/const_local_velocity_filter.h"
#include "gpu_depthmap_fusion/filter/const_global_velocity_filter.h"
#include "gpu_depthmap_fusion/filter/orientation_2d_filter.h"

class RotatedRectFilter
{
public:
    RotatedRectFilter();
    RotatedRectFilter(const cv::RotatedRect& rrect);
    void filter(double dt, const cv::RotatedRect& rrect);
    cv::RotatedRect rrect;
protected:
    // ConstLocalVelocityFilter<Orientation2DFilter<double>, Orientation2DFilter<double>::TNumOrientation, double, 2> kinematic_filter;
    ConstGlobalVelocityFilter<double, 2> kinematic_filter;
    Orientation2DFilter<double> orientation_filter;
    GainFilter<double, 2> size_filter;
};


RotatedRectFilter::RotatedRectFilter()
{
    double ref_dt = 0.1;
    orientation_filter = Orientation2DFilter<double>(
        // value
        // 0, ref_dt, // high gain means trust the prediction, low gain means trust the last values
        // 1, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
        1.0, ref_dt,  // high gain means trust the prediction, low gain means trust the last values
        0.5, ref_dt,  // high gain means trust the observation, low gain means trust the predicted values
        // value
        // 1, ref_dt, // high gain means trust the prediction, low gain means trust the last values
        // 0.2, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
        // 0.4, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
        // velocity
        1.0, ref_dt,  // high gain means trust the prediction, low gain means trust the last values
        0.5, ref_dt,  // high gain means trust the observation, low gain means trust the predicted values
        // 0 // no rotational wrap 
        M_PI/2 // rotational wrap of 90°
    );
    kinematic_filter = ConstGlobalVelocityFilter<double, 2>(
        // // value
        // 0, ref_dt, // high gain means trust the prediction, low gain means trust the last values
        // 1, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
        // // velocity
        // 1.0, ref_dt,  // high gain means trust the prediction, low gain means trust the last values
        // 0.0, ref_dt,  // high gain means trust the observation, low gain means trust the predicted values
        
        // value
        1, ref_dt, // high gain means trust the prediction, low gain means trust the last values
        0.3, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
        // velocity
        1.0, ref_dt,  // high gain means trust the velocity prediction, low gain means trust the last values
        0.0, ref_dt  // high gain means trust the velocity observation, low gain means trust the predicted values
        // 0.05, ref_dt,  // high gain means trust the velocity prediction, low gain means trust the last values
        // 0.9, ref_dt  // high gain means trust the velocity observation, low gain means trust the predicted values
        
        // // value
        // 1, ref_dt, // high gain means trust the prediction, low gain means trust the last values
        // 0.4, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
        // // velocity
        // 1.0, ref_dt,  // high gain means trust the prediction, low gain means trust the last values
        // 0.0, ref_dt,  // high gain means trust the observation, low gain means trust the predicted values
        
        // // value
        // 1, ref_dt, // high gain means trust the prediction, low gain means trust the last values
        // 0.4, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
        // // velocity
        // 0.0, ref_dt,  // high gain means trust the prediction, low gain means trust the last values
        // 1.0, ref_dt,  // high gain means trust the observation, low gain means trust the predicted values
    );
    // kinematic_filter = ConstLocalVelocityFilter<Orientation2DFilter<double>, Orientation2DFilter<double>::TNumOrientation, double, 2>(
    //     // // value
    //     // 0, ref_dt, // high gain means trust the prediction, low gain means trust the last values
    //     // 1, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
    //     // // velocity
    //     // 1.0, ref_dt,  // high gain means trust the prediction, low gain means trust the last values
    //     // 0.0, ref_dt,  // high gain means trust the observation, low gain means trust the predicted values
        
    //     // value
    //     1, ref_dt, // high gain means trust the prediction, low gain means trust the last values
    //     0.5, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
    //     // velocity
    //     0.1, ref_dt,  // high gain means trust the velocity prediction, low gain means trust the last values
    //     0.5, ref_dt  // high gain means trust the velocity observation, low gain means trust the predicted values
        
    //     // // value
    //     // 1, ref_dt, // high gain means trust the prediction, low gain means trust the last values
    //     // 0.4, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
    //     // // velocity
    //     // 1.0, ref_dt,  // high gain means trust the prediction, low gain means trust the last values
    //     // 0.0, ref_dt,  // high gain means trust the observation, low gain means trust the predicted values
        
    //     // // value
    //     // 1, ref_dt, // high gain means trust the prediction, low gain means trust the last values
    //     // 0.4, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
    //     // // velocity
    //     // 0.0, ref_dt,  // high gain means trust the prediction, low gain means trust the last values
    //     // 1.0, ref_dt,  // high gain means trust the observation, low gain means trust the predicted values
    //     orientation_filter
    // );
    size_filter = GainFilter<double, 2>(
        0.2, ref_dt
        // 0.4, ref_dt
    );
}

RotatedRectFilter::RotatedRectFilter(const cv::RotatedRect& rrect)
    : RotatedRectFilter()
{
    filter(1, rrect);
}

void RotatedRectFilter::filter(double dt, const cv::RotatedRect& rrect)
{
    // std::cout << "filtered rrect" << std::endl;
    // std::cout << "dt   " << dt << std::endl;

    double pos[2];
    double angle;
    double size[2];

    copy<double, 2>(kinematic_filter.values, pos);
    // copy<double, 1>(kinematic_filter.orientation, &angle);
    copy<double, 1>(orientation_filter.orientation, &angle);
    copy<double, 2>(size_filter.values, size);

    // std::cout << "  before filtering" << std::endl;
    // std::cout << "    pos   " << pos[0] << " " << pos[1] << std::endl;
    // std::cout << "    size  " << size[0] << " " << size[1] << std::endl;
    // std::cout << "    angle " << angle << std::endl;


    pos[0] = rrect.center.x;
    pos[1] = rrect.center.y;
    angle = rrect.angle * M_PI / 180;
    size[0] = rrect.size.width;
    size[1] = rrect.size.height;

    // std::cout << "  filter with new" << std::endl;
    // std::cout << "    pos   " << pos[0] << " " << pos[1] << std::endl;
    // std::cout << "    size  " << size[0] << " " << size[1] << std::endl;
    // std::cout << "    angle " << angle << std::endl;

    // kinematic_filter.observe(dt, pos, &angle);
    kinematic_filter.observe(dt, pos);
    orientation_filter.correct(dt, &angle); 

    size_filter.filter(dt, size);

    copy<double, 2>(kinematic_filter.values, pos);
    // copy<double, 1>(kinematic_filter.orientation, &angle);
    copy<double, 1>(orientation_filter.orientation, &angle);
    copy<double, 2>(size_filter.values, size);

    // std::cout << "  after filtering" << std::endl;
    // std::cout << "    pos   " << pos[0] << " " << pos[1] << std::endl;
    // std::cout << "    size  " << size[0] << " " << size[1] << std::endl;
    // std::cout << "    angle " << angle << std::endl;

    this->rrect = cv::RotatedRect(
        cv::Point2f(pos[0], pos[1]),
        cv::Size2f(size[0], size[1]),
        angle * 180 / M_PI
    );
}
