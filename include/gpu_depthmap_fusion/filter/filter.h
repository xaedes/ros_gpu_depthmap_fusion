#pragma once

#include <math.h>


typedef unsigned int uint;


template<typename T, uint TNum>
void copy(const T* source, T* destination)
{
    for (int i = 0; i < TNum; ++i)
    {
        destination[i] = source[i];
    }
}


template<typename T, uint TNum>
class GainFilter
{
public:
    GainFilter()
        : gain(0.5)
        , reference_dt(1)
        , has_values(false)
    {}
    GainFilter(T gain, double reference_dt)
        : gain(gain)
        , reference_dt(reference_dt)
        , has_values(false)
    {}

    GainFilter& filter(const T* new_values)
    {
        if (has_values)
        {
            for (uint i = 0; i < TNum; ++i)
            {
                values[i] = new_values[i] * gain + (1 - gain) * values[i];
            }
        }
        else
        {
            copy<T,TNum>(new_values, values);
            has_values = true;
        }
        return *this;
    }

    GainFilter& filter(double dt, const T* new_values)
    {
        if (has_values)
        {
            T the_gain = gain_for_dt(dt);
            // std::cout << "the_gain " << the_gain << std::endl;
            for (uint i = 0; i < TNum; ++i)
            {
                values[i] = new_values[i] * the_gain + (1 - the_gain) * values[i];
            }
        }
        else
        {
            copy<T,TNum>(new_values, values);
            has_values = true;
        }
        return *this;
    }

    T gain_for_dt(double dt)
    {
        if (abs(gain) < 1e-9) return 0;
        else {

            // this formulare is when filter equation y_new = x_new * (1-gain) + (gain) * y_old
            // T denom = (((reference_dt * (1 - gain)) / gain) + dt); 
             
            // this formulare is when filter equation y_new = x_new * (gain) + (1-gain) * y_old
            T denom = (reference_dt / gain) + dt - reference_dt;

            if (abs(denom) < 1e-9) return 1;
            else return dt / denom;
        }
    }

    T values[TNum];
    T gain;

    bool has_values;
    double reference_dt; // time always as double
};



template<typename T, uint TNum>
class ObservePredictFilter
{
public:
    ObservePredictFilter()
        : ObservePredictFilter(0.5, 1, 0.5, 1)
    {}
    ObservePredictFilter(T prediction_gain, double prediction_gain_dt, T correction_gain, double correction_gain_dt)
        : prediction_filter(prediction_gain, prediction_gain_dt)
        , correction_filter(correction_gain, correction_gain_dt)
        , has_values(false)
    {
        for (int i = 0; i < TNum; ++i)
        {
            values[i] = 0;
        }
    }

    void correct(double dt, const T* observed)
    {
        if (has_values)
        {
            copy<T,TNum>(values, correction_filter.values);
            correction_filter.filter(dt, observed);
            copy<T,TNum>(correction_filter.values, values);
        }
        else
        {
            has_values = true;
            copy<T,TNum>(observed, values);
            copy<T,TNum>(values, correction_filter.values);
            copy<T,TNum>(values, prediction_filter.values);
            correction_filter.has_values = true;
            prediction_filter.has_values = true;
        }
    }
    void predict(double dt, const T* prediction)
    {
        if (has_values)
        {
            copy<T,TNum>(values, prediction_filter.values);
            prediction_filter.filter(dt, prediction);
            copy<T,TNum>(prediction_filter.values, values);
        }
        else
        {
            has_values = true;
            copy<T,TNum>(prediction, values);
            copy<T,TNum>(values, correction_filter.values);
            copy<T,TNum>(values, prediction_filter.values);
            correction_filter.has_values = true;
            prediction_filter.has_values = true;
        }
    }

    T values[TNum];
    bool has_values;

    GainFilter<T, TNum> prediction_filter;
    GainFilter<T, TNum> correction_filter;
}; 
