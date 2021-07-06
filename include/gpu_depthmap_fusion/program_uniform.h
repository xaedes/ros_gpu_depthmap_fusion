#pragma once
#include <glad/glad.h>
#include <string>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>

template <typename T>
class ProgramUniform
{
public:
    ProgramUniform() {}
    void init(GLuint glProgram, const std::string& name)
    {
        m_glProgram = glProgram;
        m_name = name;
        m_nameZeroed = m_name + '\0';
        m_loc = glGetUniformLocation(m_glProgram, m_nameZeroed.c_str());
        if (m_enableDebugOutput)
        {
            std::cout << "location for " << m_name << " is " << m_loc << std::endl;
            if (m_loc < 0)
            {
                std::cout << "location for uniform " << m_name << " not found!" << std::endl;
            }
        }
    }
    void set(const T& value);
    inline const T& get() const { return m_value; }

public:
    T m_value;
    GLuint m_glProgram;
    GLint m_loc = -1;
    std::string m_name;
    std::string m_nameZeroed;
    bool m_enableDebugOutput = false;
};


template<> void ProgramUniform<float>::set(const float& value)
{
    m_value = value;
    if (m_loc != -1) 
    {
        glProgramUniform1f(m_glProgram, m_loc, m_value);
    }
    else if (m_enableDebugOutput)
    {
        std::cout << "location for uniform " << m_name << " not found!" << std::endl;
    }
}
template<> void ProgramUniform<uint>::set(const uint& value)
{
    m_value = value;
    if (m_loc != -1) 
    {
        glProgramUniform1ui(m_glProgram, m_loc, m_value);
    }
    else
 if (m_enableDebugOutput)    {
        std::cout << "location for uniform " << m_name << " not found!" << std::endl;
    }
}
template<> void ProgramUniform<int>::set(const int& value)
{
    m_value = value;
    if (m_loc != -1) 
    {
        glProgramUniform1i(m_glProgram, m_loc, m_value);
    }
    else if (m_enableDebugOutput)
    {
        std::cout << "location for uniform " << m_name << " not found!" << std::endl;
    }
}
template<> void ProgramUniform<glm::vec2>::set(const glm::vec2& value)
{
    m_value = value;
    if (m_loc != -1) 
    {
        glProgramUniform2fv(m_glProgram, m_loc, 1, &m_value[0]);
    }
    else if (m_enableDebugOutput)
    {
        std::cout << "location for uniform " << m_name << " not found!" << std::endl;
    }
}
template<> void ProgramUniform<glm::vec3>::set(const glm::vec3& value)
{
    m_value = value;
    if (m_loc != -1) 
    {
        glProgramUniform3fv(m_glProgram, m_loc, 1, &m_value[0]);
    }
    else if (m_enableDebugOutput)
    {
        std::cout << "location for uniform " << m_name << " not found!" << std::endl;
    }
}
template<> void ProgramUniform<glm::vec4>::set(const glm::vec4& value)
{
    m_value = value;
    if (m_loc != -1) 
    {
        glProgramUniform4fv(m_glProgram, m_loc, 1, &m_value[0]);
    }
    else if (m_enableDebugOutput)
    {
        std::cout << "location for uniform " << m_name << " not found!" << std::endl;
    }
}
template<> void ProgramUniform<glm::ivec2>::set(const glm::ivec2& value)
{
    m_value = value;
    if (m_loc != -1) 
    {
        glProgramUniform2iv(m_glProgram, m_loc, 1, &m_value[0]);
    }
    else if (m_enableDebugOutput)
    {
        std::cout << "location for uniform " << m_name << " not found!" << std::endl;
    }
}
template<> void ProgramUniform<glm::ivec3>::set(const glm::ivec3& value)
{
    m_value = value;
    if (m_loc != -1) 
    {
        glProgramUniform3iv(m_glProgram, m_loc, 1, &m_value[0]);
    }
    else if (m_enableDebugOutput)
    {
        std::cout << "location for uniform " << m_name << " not found!" << std::endl;
    }
}
template<> void ProgramUniform<glm::ivec4>::set(const glm::ivec4& value)
{
    m_value = value;
    if (m_loc != -1) 
    {
        glProgramUniform4iv(m_glProgram, m_loc, 1, &m_value[0]);
    }
    else if (m_enableDebugOutput)
    {
        std::cout << "location for uniform " << m_name << " not found!" << std::endl;
    }
}
template<> void ProgramUniform<glm::uvec2>::set(const glm::uvec2& value)
{
    m_value = value;
    if (m_loc != -1) 
    {
        glProgramUniform2uiv(m_glProgram, m_loc, 1, &m_value[0]);
    }
    else if (m_enableDebugOutput)
    {
        std::cout << "location for uniform " << m_name << " not found!" << std::endl;
    }
}
template<> void ProgramUniform<glm::uvec3>::set(const glm::uvec3& value)
{
    m_value = value;
    if (m_loc != -1) 
    {
        glProgramUniform3uiv(m_glProgram, m_loc, 1, &m_value[0]);
    }
    else if (m_enableDebugOutput)
    {
        std::cout << "location for uniform " << m_name << " not found!" << std::endl;
    }
}
template<> void ProgramUniform<glm::uvec4>::set(const glm::uvec4& value)
{
    m_value = value;
    if (m_loc != -1) 
    {
        glProgramUniform4uiv(m_glProgram, m_loc, 1, &m_value[0]);
    }
    else if (m_enableDebugOutput)
    {
        std::cout << "location for uniform " << m_name << " not found!" << std::endl;
    }
}
// template<> void ProgramUniform<glm::mat4>::set(const glm::mat4& value)
// {
//     m_value = value;
//     if (m_loc != -1) 
//     {
//         bool transpose = false;
//         glProgramUniformMatrix4fv(m_glProgram, m_loc, 16, transpose, &m_value[0]);
//     }
//     else if (m_enableDebugOutput)
//     {
//         std::cout << "location for uniform " << m_name << " not found!" << std::endl;
//     }
// }
template<> void ProgramUniform<cv::Matx44f>::set(const cv::Matx44f& value)
{
    m_value = value;
    if (m_loc != -1) 
    {
        bool transpose = false;
        glProgramUniformMatrix4fv(m_glProgram, m_loc, 1, transpose, &m_value(0,0));
    }
    else if (m_enableDebugOutput)
    {
        std::cout << "location for uniform " << m_name << " not found!" << std::endl;
    }
}
