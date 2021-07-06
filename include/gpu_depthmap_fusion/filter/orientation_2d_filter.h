#pragma once

#include "filter.h"
#include "wrap_pi.h"
#include "const_global_velocity_filter.h"
#include "rotate_vector.h"

template<typename T>
class Orientation2DFilter
{
public:
    static const uint TNumOrientation = 1;
    Orientation2DFilter()
        : Orientation2DFilter(
            0.5, 1,
            0.5, 1,
            0.5, 1,
            0.5, 1,
            0
        )
    {}
    Orientation2DFilter(
        T      value_prediction_gain,             // high gain means trust the prediction, low gain means trust the last values
        double value_prediction_gain_dt,
        T      value_correction_gain,             // high gain means trust the observation, low gain means trust the predicted values
        double value_correction_gain_dt,
        T      velocity_prediction_gain,          // high gain means trust the prediction, low gain means trust the last values
        double velocity_prediction_gain_dt,
        T      velocity_correction_gain,          // high gain means trust the observation, low gain means trust the predicted values
        double velocity_correction_gain_dt,
        T rotation_wrap
    )
        : filter(
            value_prediction_gain, value_prediction_gain_dt, 
            value_correction_gain, value_correction_gain_dt, 
            velocity_prediction_gain, velocity_prediction_gain_dt, 
            velocity_correction_gain, velocity_correction_gain_dt)
        , rotation_wrap(rotation_wrap)
    {
        for (int i=0; i<TNumOrientation; i++)
        {
            orientation[i] = 0;
            turnrate[i] = 0;
        }
    }

    void observe(double dt, const T* observed_values)
    {
        predict(dt);
        correct(dt, observed_values);
    }
    
    void correct(double dt, const T* observed_values)
    {
        T unwrapped_values[TNumOrientation];
        if (filter.has_last_measurement)
        {
            // unwrap roll pitch yaw
            for (int i=0; i<TNumOrientation; i++)
            {
                auto diff = angleDiff(filter.last_measurement[i], observed_values[i]);
                if (rotation_wrap != 0)
                {
                    diff = -rotation_wrap/2 + fmod(diff + rotation_wrap/2, rotation_wrap);
                }
                
                unwrapped_values[i] = filter.last_measurement[i] + diff;
                // unwrapped_values[i] = wrapToPiSeq(filter.last_measurement[i], observed_values[i]);
            }
        }
        else
        {
            for (int i=0; i<TNumOrientation; i++)
                unwrapped_values[i] = observed_values[i];
        }
        filter.correct(dt, unwrapped_values);
        copy<T, TNumOrientation>(filter.values, orientation);
        copy<T, TNumOrientation>(filter.velocity, turnrate);
    }

    void predict(double dt)
    {
        filter.predict(dt);
        copy<T, TNumOrientation>(filter.values, orientation);
        copy<T, TNumOrientation>(filter.velocity, turnrate);
    }

    void rotate_vector(const T* vec_in, T* vec_out) const
    {
        rotate_vector(orientation, vec_in, vec_out);
    }

    static void rotate_vector(const T* orientation, const T* vec_in, T* vec_out)
    {
        T mat[2*2];
        to_matrix(orientation, mat);
        ::rotate_vector<T, 2>(mat, vec_in, vec_out);
    }

    void rotate_vector_inv(const T* vec_in, T* vec_out) const
    {
        rotate_vector_inv(orientation, vec_in, vec_out);
    }

    static void rotate_vector_inv(const T* orientation, const T* vec_in, T* vec_out)
    {
        T mat[2*2];
        to_matrix(orientation, mat);
        ::rotate_vector_inv<T, 2>(mat, vec_in, vec_out);
    }

    void to_matrix(T* mat_out) const
    {
        to_matrix(orientation, mat_out);
    }
    static void to_matrix(const T* orientation, T* mat_out)
    {

        const int W = 2;
        T _cos = cos(orientation[0]);
        T _sin = sin(orientation[0]);
        mat_out[W*0 + 0] = _cos;
        mat_out[W*0 + 1] = _sin;

        mat_out[W*1 + 0] = _sin;
        mat_out[W*1 + 1] = _cos;
    }

    T orientation[TNumOrientation];
    T turnrate[TNumOrientation];
    T rotation_wrap;

    ConstGlobalVelocityFilter<T, TNumOrientation> filter;
};

