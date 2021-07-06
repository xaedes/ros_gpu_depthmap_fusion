
#include <iostream>
#include <vector>

#include "const_global_velocity_filter.h"
#include "const_local_velocity_filter.h"
#include "roll_pitch_yaw_filter.h"

template<typename T, uint TNum>
void print(const ConstGlobalVelocityFilter<T, TNum>& filter)
{
    std::cout << "filter.values";
    for (int i = 0; i < TNum; ++i)
    {
        std::cout << " " << filter.values[i];
    }
    std::cout << std::endl;
    std::cout << "filter.velocity";
    for (int i = 0; i < TNum; ++i)
    {
        std::cout << " " << filter.velocity[i];
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

template<typename TOrientationFilter, uint TNumOrientation, typename T, uint TNum>
void print(const ConstLocalVelocityFilter<TOrientationFilter, TNumOrientation, T, TNum>& filter)
{
    std::cout << "filter.pos";
    for (int i = 0; i < TNum; ++i)
    {
        std::cout << " " << filter.pos[i];
    }
    std::cout << std::endl;
    std::cout << "filter.velocity";
    for (int i = 0; i < TNum; ++i)
    {
        std::cout << " " << filter.velocity[i];
    }
    std::cout << std::endl;
    std::cout << "filter.orientation";
    for (int i = 0; i < TNum; ++i)
    {
        std::cout << " " << filter.orientation[i];
    }
    std::cout << std::endl;
    std::cout << "filter.turnrate";
    for (int i = 0; i < TNum; ++i)
    {
        std::cout << " " << filter.turnrate[i];
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

void demo_ConstGlobalVelocityFilter(double dt, const std::vector<double>& positions, int num_extrapolate)
{
    double ref_dt = 0.1;
    ConstGlobalVelocityFilter<double, 3> filter(
        // value
        1, ref_dt, // high gain means trust the prediction, low gain means trust the last values
        0.99, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
        // velocity
        0.0, ref_dt,  // high gain means trust the prediction, low gain means trust the last values
        0.6, ref_dt  // high gain means trust the observation, low gain means trust the predicted values
    );
    int num = positions.size() / 3;
    for (int i = 0; i < num; ++i)
    {
        std::cout << " pos[" << i << "] " << positions[i * 3 + 0] << ", " << positions[i * 3 + 1] << ", " << positions[i * 3 + 2] << std::endl;
    }
    print(filter);
    for (int i = 0; i < num; ++i)
    {
        std::cout << "correct " << " dt " << dt << " values " << positions[i * 3 + 0] << ", " << positions[i * 3 + 1] << ", " << positions[i * 3 + 2] << std::endl;
        filter.correct(dt, &positions[i * 3]);
        print(filter);
    }
    for (int i = 0; i < num_extrapolate; ++i)
    {
        std::cout << "predict " << " dt " << dt << std::endl;
        filter.predict(dt);
        print(filter);
    }
}

void demo_ConstLocalVelocityFilter(double dt, const std::vector<double>& positions, const std::vector<double>& orientations, int num_extrapolate)
{
    double ref_dt = 0.1;
    RollPitchYawFilter<double> orientation_filter(
        // value
        1, ref_dt, // high gain means trust the prediction, low gain means trust the last values
        0.99, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
        // velocity
        1.0, ref_dt,  // high gain means trust the prediction, low gain means trust the last values
        0.0, ref_dt  // high gain means trust the observation, low gain means trust the predicted values
    );
    ConstLocalVelocityFilter<RollPitchYawFilter<double>, RollPitchYawFilter<double>::TNumOrientation, double, 3> filter(
        // value
        1, ref_dt, // high gain means trust the prediction, low gain means trust the last values
        0.99, ref_dt, // high gain means trust the observation, low gain means trust the predicted values
        // velocity
        0.0, ref_dt,  // high gain means trust the prediction, low gain means trust the last values
        0.6, ref_dt,  // high gain means trust the observation, low gain means trust the predicted values
        orientation_filter
    );
    int num = positions.size() / 3;
    for (int i = 0; i < num; ++i)
    {
        std::cout << " pos[" << i << "] " << positions[i * 3 + 0] << ", " << positions[i * 3 + 1] << ", " << positions[i * 3 + 2];
        std::cout << " orientation[" << i << "] " << orientations[i * 3 + 0] << ", " << orientations[i * 3 + 1] << ", " << orientations[i * 3 + 2] << std::endl;
    }
    print(filter);
    for (int i = 0; i < num; ++i)
    {
        std::cout << "correct " << " dt " << dt;
        std::cout << " pos " << positions[i * 3 + 0] << ", " << positions[i * 3 + 1] << ", " << positions[i * 3 + 2] << std::endl;
        std::cout << " orientation " << orientations[i * 3 + 0] << ", " << orientations[i * 3 + 1] << ", " << orientations[i * 3 + 2] << std::endl;
        filter.correct(dt, &positions[i * 3], &orientations[i * 3]);
        print(filter);
    }
    for (int i = 0; i < num_extrapolate; ++i)
    {
        std::cout << "predict " << " dt " << dt << std::endl;
        filter.predict(dt);
        print(filter);
    }
}

int main()
{
    demo_ConstGlobalVelocityFilter(
        0.1, {
        2, 2, 2,
        1, 1, 1,
        3, 3, 3,
        }, 3);
    const double D2R = M_PI / 180;
    demo_ConstLocalVelocityFilter(
        0.1, {
        2, 0, 0,
        1, 0, 0,
        3, 0, 0,
        }, {
        0, 0, 0,
        0, 0, 0,
        0, 0, 45*D2R,
        }, 3);
}
