#pragma once

#include "filter.h"
#include "wrap_pi.h"
#include "const_global_velocity_filter.h"
#include "rotate_vector.h"

template<typename T>
class RollPitchYawFilter
{
public:
    static const uint TNumOrientation = 3;
    RollPitchYawFilter()
        : RollPitchYawFilter(
            0.5, 1,
            0.5, 1,
            0.5, 1,
            0.5, 1
        )
    {}
    RollPitchYawFilter(
        T      value_prediction_gain,             // high gain means trust the prediction, low gain means trust the last values
        double value_prediction_gain_dt,
        T      value_correction_gain,             // high gain means trust the observation, low gain means trust the predicted values
        double value_correction_gain_dt,
        T      velocity_prediction_gain,          // high gain means trust the prediction, low gain means trust the last values
        double velocity_prediction_gain_dt,
        T      velocity_correction_gain,          // high gain means trust the observation, low gain means trust the predicted values
        double velocity_correction_gain_dt
    )
        : filter(
            value_prediction_gain, value_prediction_gain_dt, 
            value_correction_gain, value_correction_gain_dt, 
            velocity_prediction_gain, velocity_prediction_gain_dt, 
            velocity_correction_gain, velocity_correction_gain_dt)
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
                unwrapped_values[i] = wrapToPiSeq(filter.last_measurement[i], observed_values[i]);
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
        T mat[3*3];
        to_matrix(orientation, mat);
        ::rotate_vector<T, 3>(mat, vec_in, vec_out);
    }

    void rotate_vector_inv(const T* vec_in, T* vec_out) const
    {
        rotate_vector_inv(orientation, vec_in, vec_out);
    }

    static void rotate_vector_inv(const T* orientation, const T* vec_in, T* vec_out)
    {
        T mat[3*3];
        to_matrix(orientation, mat);
        ::rotate_vector_inv<T, 3>(mat, vec_in, vec_out);
    }

    void to_matrix(T* mat_out) const
    {
        to_matrix(orientation, mat_out);
    }
    static void to_matrix(const T* orientation, T* mat_out)
    {

        const int W = 3;
        T _cos[TNumOrientation];
        T _sin[TNumOrientation];
        for (int i = 0; i <TNumOrientation; i++)
        {
            _cos[i] = cos(orientation[i]);
            _sin[i] = sin(orientation[i]);
        }
        // mat = rotate_z(yaw) * rotate_y(pitch) * rotate_x(roll)
        // http://planning.cs.uiuc.edu/node102.html
        const int ROLL = 0;
        const int PITCH = 1;
        const int YAW = 2;
        mat_out[W*0 + 0] = _cos[YAW]*_cos[PITCH];
        mat_out[W*0 + 1] = _cos[YAW]*_sin[PITCH]*_sin[ROLL] - _sin[YAW]*_cos[ROLL];
        mat_out[W*0 + 2] = _cos[YAW]*_sin[PITCH]*_cos[ROLL] + _sin[YAW]*_sin[ROLL];

        mat_out[W*1 + 0] = _sin[YAW]*_cos[PITCH];
        mat_out[W*1 + 1] = _sin[YAW]*_sin[PITCH]*_sin[ROLL] + _cos[YAW]*_cos[ROLL];
        mat_out[W*1 + 2] = _sin[YAW]*_sin[PITCH]*_cos[ROLL] - _cos[YAW]*_sin[ROLL];

        mat_out[W*2 + 0] = - _sin[PITCH];
        mat_out[W*2 + 1] = _cos[PITCH]*_sin[ROLL];
        mat_out[W*2 + 2] = _cos[PITCH]*_cos[ROLL];
    }

    T orientation[TNumOrientation];
    T turnrate[TNumOrientation];

    ConstGlobalVelocityFilter<T, TNumOrientation> filter;
};

