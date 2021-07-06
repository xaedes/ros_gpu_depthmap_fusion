#pragma once

#include "filter.h"

template<typename T, uint TNum>
class ConstGlobalVelocityFilter
{
public:
    ConstGlobalVelocityFilter() {}
    ConstGlobalVelocityFilter(
        T value_prediction_gain,             // high gain means trust the prediction, low gain means trust the last values
        double value_prediction_gain_dt,
        T value_correction_gain,             // high gain means trust the observation, low gain means trust the predicted values
        double value_correction_gain_dt,
        T velocity_prediction_gain,          // high gain means trust the prediction, low gain means trust the last values
        double velocity_prediction_gain_dt,
        T velocity_correction_gain,          // high gain means trust the observation, low gain means trust the predicted values
        double velocity_correction_gain_dt
    )
        : value_filter(
            value_prediction_gain, value_prediction_gain_dt, 
            value_correction_gain, value_correction_gain_dt)
        , velocity_filter(
            velocity_prediction_gain, velocity_prediction_gain_dt, 
            velocity_correction_gain, velocity_correction_gain_dt)
        , has_last_measurement(false)
    {
        for (int i = 0; i < TNum; ++i)
        {
            // you can set another predicted velocity after instantiating
            predicted_velocity[i] = 0;
            values[i] = 0;
            velocity[i] = 0;
        }
    }

    void observe(double dt, const T* observed_values)
    {
        predict(dt);
        correct(dt, observed_values);
    }

    void correct(double dt, const T* observed_values)
    {
        T observed_velocity[TNum];
        if (has_last_measurement && abs(dt) > 1e-6)
        {
            for (int i = 0; i < TNum; ++i)
            {
                observed_velocity[i] = (observed_values[i] - last_measurement[i]) / dt;
            }
            // high gain means trust the observation, low gain means trust the predicted values
            velocity_filter.correct(dt, observed_velocity);
            copy<T, TNum>(velocity_filter.values, velocity);
        }
        // high gain means trust the observation, low gain means trust the predicted values
        value_filter.correct(dt, observed_values);
        copy<T, TNum>(value_filter.values, values);
        copy<T, TNum>(observed_values, last_measurement);
        has_last_measurement = true;

    }
    void predict(double dt)
    {
        if (has_last_measurement)
        {
            T predicted[TNum];
            for (uint i = 0; i < TNum; ++i)
            {
                predicted[i] = values[i] + velocity[i] * dt;
            }
            // mix between old estimate and predicted
            // high gain means trust the prediction, low gain means trust the last values
            value_filter.predict(dt, predicted);
            velocity_filter.predict(dt, predicted_velocity);
            copy<T, TNum>(value_filter.values, values);
            copy<T, TNum>(velocity_filter.values, velocity);
        }
    }
    T values[TNum];
    T velocity[TNum];

    ObservePredictFilter<T, TNum> value_filter;
    ObservePredictFilter<T, TNum> velocity_filter;

    T last_measurement[TNum];
    bool has_last_measurement;
    T predicted_velocity[TNum];

};

