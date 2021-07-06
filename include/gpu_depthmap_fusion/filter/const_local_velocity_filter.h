#pragma once

#include "filter.h"

template<typename TOrientationFilter, uint TNumOrientation, typename T, uint TNum>
class ConstLocalVelocityFilter
{
public:
    ConstLocalVelocityFilter() {}
    ConstLocalVelocityFilter(
        T      pos_prediction_gain,             // high gain means trust the prediction, low gain means trust the last values
        double pos_prediction_gain_dt,
        T      pos_correction_gain,             // high gain means trust the observation, low gain means trust the predicted values
        double pos_correction_gain_dt,
        T      velocity_prediction_gain,          // high gain means trust the prediction, low gain means trust the last values
        double velocity_prediction_gain_dt,
        T      velocity_correction_gain,          // high gain means trust the observation, low gain means trust the predicted values
        double velocity_correction_gain_dt,
        const TOrientationFilter& orientation_filter
    )
        : orientation_filter(orientation_filter)
        , pos_filter(
            pos_prediction_gain, pos_prediction_gain_dt, 
            pos_correction_gain, pos_correction_gain_dt)
        , velocity_filter(
            velocity_prediction_gain, velocity_prediction_gain_dt, 
            velocity_correction_gain, velocity_correction_gain_dt)
        , has_last_pos_measurement(false)
        , has_values(false)

    {

        for (int i = 0; i < TNum; ++i)
        {
            pos[i] = 0;
            velocity[i] = 0;
            last_pos_measurement[i] = 0;
            predicted_velocity[i] = 0;
        }
        for (int i = 0; i < TNumOrientation; ++i)
        {
            orientation[i] = 0;
            turnrate[i] = 0;
        }
    }

    void observe(double dt, const T* observed_pos, const T* observed_orientation)
    {
        predict(dt);
        correct(dt, observed_pos, observed_orientation);
    }

    void correct(double dt, const T* observed_pos, const T* observed_orientation)
    {
        // if (has_values)
        // {
            T observed_global_velocity[TNum];
            T observed_local_velocity[TNum];

            if (has_last_pos_measurement && abs(dt) > 1e-6)
            {
                for (int i = 0; i < TNum; ++i)
                {
                    observed_global_velocity[i] = (observed_pos[i] - last_pos_measurement[i]) / dt;
                }
                // local velocity resolution
                orientation_filter.rotate_vector_inv(observed_global_velocity, observed_local_velocity);
                // high gain means trust the observation, low gain means trust the predicted values
                velocity_filter.correct(dt, observed_local_velocity);
                copy<T, TNum>(velocity_filter.values, velocity);
            }
            // high gain means trust the observation, low gain means trust the predicted values
            pos_filter.correct(dt, observed_pos);
            orientation_filter.correct(dt, observed_orientation); 
            copy<T, TNum>(pos_filter.values, pos);
            copy<T, TNumOrientation>(orientation_filter.orientation, orientation);
            copy<T, TNumOrientation>(orientation_filter.turnrate, turnrate);
            copy<T, TNum>(observed_pos, last_pos_measurement);
            has_last_pos_measurement = true;
            has_values = true;   
        // }
        // else
        // {
        //     copy<T, TNum>(observed_pos, pos);
        //     copy<T, TNum>(observed_orientation, orientation);
        //     has_values = true;   
        // }

    }
    void predict(double dt)
    {
        if (has_values)
        {
            orientation_filter.predict(dt);

            T global_velocity[TNum];
            orientation_filter.rotate_vector(velocity, global_velocity);

            T predicted_pos[TNum];
            for (uint i = 0; i < TNum; ++i)
            {
                predicted_pos[i] = pos[i] + global_velocity[i] * dt;
            }
            pos_filter.predict(dt, predicted_pos);
            velocity_filter.predict(dt, predicted_velocity);

            copy<T, TNum>(pos_filter.values, pos);
            copy<T, TNum>(velocity_filter.values, velocity);
            copy<T, TNumOrientation>(orientation_filter.orientation, orientation);
            copy<T, TNumOrientation>(orientation_filter.turnrate, turnrate);
        }
    }

    T pos[TNum];
    T velocity[TNum];
    T orientation[TNumOrientation];
    T turnrate[TNumOrientation];

    bool has_values;

    T last_pos_measurement[TNum];
    bool has_last_pos_measurement;
    T predicted_velocity[TNum];

    ObservePredictFilter<T, TNum> pos_filter;
    ObservePredictFilter<T, TNum> velocity_filter;
    TOrientationFilter orientation_filter;

};


