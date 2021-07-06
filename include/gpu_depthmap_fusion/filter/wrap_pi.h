#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

// Translate to equivalent angle in [0,2pi[ range
template <typename T>
T wrapTo2Pi(T rad)
{
    return fmod(rad, static_cast<T>(2 * M_PI)) + ((rad < 0) ? static_cast<T>(2 * M_PI) : 0);
}



// Translate to equivalent angle in ]-pi,pi] range
template <typename T>
T wrapToPi(T rad)
{
    return wrapTo2Pi<T>(rad + static_cast<T>(M_PI)) - static_cast<T>(M_PI);
}



// Translate rad_now to equivalent angle so the jump from rad_before won't be greater than |M_PI|
template <typename T>
T wrapToPiSeq(T rad_before, T rad_now)
{
    rad_before = wrapToPi(rad_before);
    rad_now = wrapToPi(rad_now);
    T diff = rad_now - rad_before;
    if (diff > +static_cast<T>(M_PI)) rad_now -= static_cast<T>(2 * M_PI);
    if (diff < -static_cast<T>(M_PI)) rad_now += static_cast<T>(2 * M_PI);
    return rad_now;
}

template <typename T>
T angleDiff(T rad_before, T rad_now)
{
    T rad_now_ = wrapToPiSeq<T>(rad_before, rad_now);
    return wrapToPi<T>(rad_now_ - rad_before);
}
