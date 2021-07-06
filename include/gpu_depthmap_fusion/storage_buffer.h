#pragma once

#include <glad/glad.h>

template<typename T>
class StorageBuffer 
{
public:
    StorageBuffer(GLenum usage)
        : m_bufferBase(0), m_glBuf(-1), m_usage(usage), m_glBufSize(0), m_numItems(1024)
    {}
    StorageBuffer()
        : StorageBuffer(GL_STATIC_DRAW)
    {}
    void init() 
    {
        init(m_usage, m_numItems);
    }
    void init(GLenum usage, int numItems)
    {
        m_usage = usage;
        glGenBuffers(1, &m_glBuf);
        resize(numItems);
    }
    void resize(int numItems)
    {
        m_numItems = numItems;
        int newBufSize = sizeof(T) * m_numItems;
        // align size to 4 bytes
        if (newBufSize % 4 != 0)
        {
            newBufSize += 4 - (newBufSize % 4);
        }
        
        if (newBufSize > m_glBufSize)
        {
            m_glBufSize = newBufSize;
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_glBuf);
            // std::vector<T> data(m_numItems);
            // glBufferData(GL_SHADER_STORAGE_BUFFER, m_glBufSize, data.data(), m_usage);
            glBufferData(GL_SHADER_STORAGE_BUFFER, m_glBufSize, NULL, m_usage);
        }
        else
        {
            m_glBufSize = newBufSize;
        }
    }
    StorageBuffer<T>& bind()
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_glBuf);
        return *this;
    }
    StorageBuffer<T>& upload(const void* data)
    {
        upload(data, 0, m_numItems);
        return *this;
    }
    StorageBuffer<T>& upload(const void* data, int start, int num)
    {
        if (m_autoBind) bind();
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(T)*start, sizeof(T)*(num), data);
        return *this;
    }
    StorageBuffer<T>& download(void* data)
    {
        download(data, 0, m_numItems);
        return *this;
    }
    StorageBuffer<T>& download(void* data, int start, int num)
    {
        if (m_autoBind) bind();
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(T)*start, sizeof(T)*(num), data);
        return *this;
    }
    GLuint bufferBase()
    {
        return m_bufferBase;
    }
    StorageBuffer<T>& bufferBase(GLuint value)
    {
        m_bufferBase = value;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, m_bufferBase, m_glBuf);
        return *this;
    }
    void* map_ro() { return map(GL_READ_ONLY); }
    void* map_wo() { return map(GL_WRITE_ONLY); }
    void* map_rw() { return map(GL_READ_WRITE); }
    void* map(GLenum access)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_glBuf);
        return glMapBuffer(GL_SHADER_STORAGE_BUFFER, access);
    }
    
    void* mapr_ro() { return mapr_ro(0, m_numItems); }
    void* mapr_wo() { return mapr_wo(0, m_numItems); }
    void* mapr_rw() { return mapr_rw(0, m_numItems); }
    void* mapr(GLbitfield access) { return mapr(0, m_numItems, access); }

    void* mapr_ro(int start, int num) { return mapr(start, num, GL_MAP_READ_BIT); }
    void* mapr_wo(int start, int num) { return mapr(start, num, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT); }
    void* mapr_rw(int start, int num) { return mapr(start, num, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT); }
    void* mapr(int start, int num, GLbitfield access)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_glBuf);
        return glMapBufferRange(GL_SHADER_STORAGE_BUFFER, sizeof(T)*start, sizeof(T)*(num), access);
    }
        // return glMapBufferRange(GL_SHADER_STORAGE_BUFFER, access);
    void unmap()
    {
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    int size() const { return m_numItems; }
    GLuint glBuf() const { return m_glBuf; }
    
    GLuint m_bufferBase;
    GLuint m_glBuf;
    GLenum m_usage;
    int m_numItems;
    int m_glBufSize;
    bool m_autoBind = false;
};
