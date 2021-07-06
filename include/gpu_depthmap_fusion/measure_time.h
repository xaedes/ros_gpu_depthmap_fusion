#pragma once

#include <chrono>
#include <vector>

template<typename TDurationTime>
class MeasureTime_
{
public:
    MeasureTime_() : gain(0.9) {}
    MeasureTime_(float gain) : gain(gain) {}
    float gain;
    std::vector<std::string> sections;
    inline void clear()
    {}
    inline void beginFrame()
    {
        m_lastFrameDurations.clear();
    }
    inline void begin()
    {
        m_timePoints.clear();
        measure();
    }
    inline void nextSection()
    {
        measure();
    }
    inline void measure()
    {
        m_timePoints.push_back(
            std::chrono::high_resolution_clock::now()
        );
    }
    inline void end()
    {
        measure();
        computeDurations();
        addFrameDurations();
        
    }
    inline void endFrame()
    {
        filterFrameDurations();
    }
    inline const std::vector<float>& durations() const
    {
        return m_durations;
    }
    inline const std::vector<float>& lastDurations() const
    {
        return m_lastDurations;
    }
    inline const std::vector<float>& timePoints() const
    {
        return m_timePoints;
    }
    void print(const std::string& title)
    {
        std::cout << title << std::endl;
        for (int i = 1; i < m_durations.size(); ++i)
        {
            std::cout << "duration" << i << " ";
            if (m_numMeasurements[i] > 0)
            {
                std::cout << m_durations[i] << " ";
                if (i-1 < sections.size())
                {
                    std::cout << sections[i-1];
                }
                std::cout << std::endl;
            }
        }
        if (m_numMeasurements.size() > 0)
        if (m_numMeasurements[0] > 0)
        {
            std::cout << "total duration" << " ";
            std::cout << m_durations[0] << " ";
            std::cout << std::endl;
        }        
        std::cout << std::endl;
    }
protected:
    inline void computeDurations()
    {
        // idx 0 duration is total duration
        m_lastDurations.resize(m_timePoints.size());
        m_lastDurations[0] = std::chrono::duration_cast<TDurationTime>( 
            m_timePoints[m_timePoints.size()-1] - m_timePoints[0]
        ).count();
        for (size_t i = 1; i < m_timePoints.size(); i++)
        {
            m_lastDurations[i] = std::chrono::duration_cast<TDurationTime>( 
                m_timePoints[i] - m_timePoints[i-1]
            ).count();
        }
    }
    inline void addFrameDurations()
    {
        if (m_lastFrameDurations.size() < m_lastDurations.size())
            m_lastFrameDurations.resize(m_lastDurations.size());
        for (size_t i = 0; i < m_lastDurations.size(); i++)
        {
            m_lastFrameDurations[i] += m_lastDurations[i];
        }
    }
    inline void filterFrameDurations()
    {
        if (m_durations.size() < m_lastFrameDurations.size())
            m_durations.resize(m_lastFrameDurations.size());
        if (m_numMeasurements.size() < m_lastFrameDurations.size())
            m_numMeasurements.resize(m_lastFrameDurations.size());
        for (size_t i = 0; i < m_lastFrameDurations.size(); i++)
        {
            if (m_numMeasurements[i] == 0)
            {
                m_durations[i] = m_lastFrameDurations[i];
            }
            else
            {
                m_durations[i]  = m_lastFrameDurations[i] * gain + (1-gain) * m_durations[i];
            }
            ++m_numMeasurements[i];
        }
    }
    std::vector<std::chrono::high_resolution_clock::time_point> m_timePoints;
    std::vector<float> m_lastDurations;
    std::vector<float> m_lastFrameDurations;
    std::vector<float> m_durations;
    std::vector<int> m_numMeasurements;
};

typedef MeasureTime_<std::chrono::microseconds> MeasureTime;
