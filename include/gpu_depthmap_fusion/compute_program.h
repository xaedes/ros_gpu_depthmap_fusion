#pragma once
#include <glad/glad.h>
#include <vector>
#include <utility>
#include <iostream>
#include <sstream>

#include "gpu_depthmap_fusion/load_shader.h"
#include "gpu_depthmap_fusion/replace_string.h"


void checkGLError()
{
    GLenum err;
    err = glGetError();
    // if (GL_NO_ERROR != GL_NO_ERROR)
    // {
    //     throw std::runtime_error("gl error");
    // }
    while((err = glGetError()) != GL_NO_ERROR)
    {
        std::cout << err << std::endl;
        // Process/log the error.
    }
}

class ComputeProgram 
{
public:
    typedef std::vector<std::pair<std::string,std::string>> StringReplacementsType;
    ComputeProgram()
    {}
    virtual ~ComputeProgram()
    {}

    virtual void init(const std::string& filename_shader)
    {
        init(filename_shader, {});
    }
    virtual void init(const std::string& filename_shader, const StringReplacementsType& replacements)
    {

        m_shaderCode = LoadShader(filename_shader);
        replaceStrings(replacements);

        // std::cout << "filename_shader " << filename_shader << std::endl;
        // printSourceWithLineNumbers();

        m_glComputeShader = glCreateShader(GL_COMPUTE_SHADER);
        GLchar* code = m_shaderCode.data();
        glShaderSource(m_glComputeShader, 1, &code, nullptr);
        GLchar infolog[512];
        glCompileShader(m_glComputeShader);
        glGetShaderInfoLog(m_glComputeShader, 512, nullptr, infolog);
        if (infolog[0] != '\0') 
        {
            std::cout << "filename_shader " << filename_shader << std::endl;
            printSourceWithLineNumbers();
        }
        std::cout << infolog << std::endl;
        m_glProgram = glCreateProgram();
        glAttachShader(m_glProgram, m_glComputeShader);
        glLinkProgram(m_glProgram);
        glDeleteShader(m_glComputeShader);
        glGetProgramInfoLog(m_glProgram, 512, nullptr, infolog);
        if (infolog[0] != '\0') 
        {
            std::cout << "filename_shader " << filename_shader << std::endl;
            printSourceWithLineNumbers();
        }
        std::cout << infolog << std::endl;
    }
    virtual ComputeProgram& use()
    {
        glUseProgram(m_glProgram);
        return *this;
    }
    virtual ComputeProgram& dispatch(int x, int y, int z, int gx, int gy, int gz)
    {
        ComputeProgram::dispatch(
            x/gx + ((x%gx == 0) ? 0 : 1),
            y/gy + ((y%gy == 0) ? 0 : 1),
            z/gz + ((z%gz == 0) ? 0 : 1)
        );
    }
    virtual ComputeProgram& dispatch(int x, int y, int z)
    {
        glDispatchCompute(x, y, z);
        checkGLError();
        return *this;
    }
    virtual void replaceStrings(const StringReplacementsType& replacements)
    {
        std::string source(m_shaderCode.begin(), m_shaderCode.end());
        std::vector<std::string> removes;
        std::vector<std::string> inserts;
        removes.resize(replacements.size());
        inserts.resize(replacements.size());
        for (int i = 0; i < replacements.size(); ++i)
        {
            removes.push_back("%%" + replacements[i].first + "%%");
            inserts.push_back(replacements[i].second);
        }
        replace_strings_inplace(source, removes, inserts);
        m_shaderCode.clear();
        m_shaderCode.insert(m_shaderCode.end(), source.begin(), source.end());
    }

    void printSourceWithLineNumbers()
    {
        std::string source(m_shaderCode.begin(), m_shaderCode.end());
        std::istringstream iss;
        iss.str(source);
        std::string line;
        for (int k=1; std::getline(iss, line); ++k) 
        {
            std::cout << k << "\t" << line << std::endl;
        }

    }

    GLuint getGLProgram()
    {
        return m_glProgram;
    }


protected:
    std::vector<GLchar> m_shaderCode;
    GLuint m_glProgram;
    GLuint m_glComputeShader;
};
