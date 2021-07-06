#pragma once

#include <glad/glad.h>
#include <vector>
#include <unordered_map>

class BufferIndexBindings
{
public:

    BufferIndexBindings() {}

    void init();

    struct BufferIndexBinding
    {
        GLuint glBuffer;
        GLuint bindingIndex;
    };

    struct BindingsForProgram
    {
        GLuint glProgram;
        std::vector<BufferIndexBinding> bindings;
    };


    /**
     * @brief      Make buffer binding points for buffers with increasing binding points.
     *
     * @param[in]  glBuffers  The gl buffers
     *
     * @return     vector of buffer,binding pairs
     */
    std::vector<BufferIndexBinding> make(
        const std::vector<GLuint>& glBuffers
    );

    /**
     * @brief      Add bindings for gl program. 
     *
     * @param[in]  glProgram  The gl program
     * @param[in]  bindings   The bindings
     *
     * @return     The bindings
     */
    const BindingsForProgram& add(
        GLuint glProgram,
        const std::vector<BufferIndexBinding>& bindings
    );

    /**
     * @brief      Gets bindings for the specified gl program.
     *
     * @param[in]  glProgram  The gl program
     *
     * @return     The bindings
     */
    const BindingsForProgram& get(GLuint glProgram);


    /**
     * @brief      Bind buffer bindings for the specified gl program.
     *
     * @param[in]  glProgram  The gl program
     */
    void bind(GLuint glProgram);

protected:
    GLint glMaxComputeShaderStorageBlocks;
    GLint glMaxCombinedShaderStorageBlocks;
    GLuint m_nextBindingPoint;
    std::unordered_map<GLuint, BindingsForProgram> m_bindingsPerProgram;
    std::vector<bool> m_isBound;
    std::vector<GLuint> m_boundBuffers; // content only valid if corresponding element in m_isBound is true
};

void BufferIndexBindings::init()
{
    glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS,  &glMaxComputeShaderStorageBlocks);
    glGetIntegerv(GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS, &glMaxCombinedShaderStorageBlocks);

    std::cout << "GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS  " << glMaxComputeShaderStorageBlocks << std::endl;
    std::cout << "GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS " << glMaxCombinedShaderStorageBlocks << std::endl;

    m_isBound.resize(glMaxCombinedShaderStorageBlocks); // will initialize with false
    m_boundBuffers.resize(glMaxCombinedShaderStorageBlocks);

    m_nextBindingPoint = 0;
}

std::vector<BufferIndexBindings::BufferIndexBinding> BufferIndexBindings::make(
    const std::vector<GLuint>& glBuffers
)
{
    if(glBuffers.size() > glMaxComputeShaderStorageBlocks)
    {
        throw std::runtime_error("number of buffers exceeds maximum of GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS=" + std::to_string(glMaxComputeShaderStorageBlocks));
    }
    if (m_nextBindingPoint + glBuffers.size() >= glMaxCombinedShaderStorageBlocks)
    {
        m_nextBindingPoint = 0;
    }
    std::vector<BufferIndexBinding> bindings;
    bindings.resize(glBuffers.size());
    for (int i = 0; i < glBuffers.size(); ++i)
    {
        bindings[i] = {glBuffers[i], m_nextBindingPoint+i};
    }

    m_nextBindingPoint += glBuffers.size();
    return bindings;
}

const BufferIndexBindings::BindingsForProgram& BufferIndexBindings::add(
    GLuint glProgram,
    const std::vector<BufferIndexBindings::BufferIndexBinding>& bindings
)
{
    BindingsForProgram bindingsForProgram{glProgram, bindings};
    m_bindingsPerProgram.emplace(glProgram, bindingsForProgram);
    return get(glProgram);
}

const BufferIndexBindings::BindingsForProgram& BufferIndexBindings::get(GLuint glProgram)
{
    return m_bindingsPerProgram.at(glProgram);
}

void BufferIndexBindings::bind(GLuint glProgram)
{
    const auto& bindings = get(glProgram);

    for (int i = 0; i < bindings.bindings.size(); ++i)
    {
        const auto& binding = bindings.bindings[i];
        // only bind if any buffer is not already bound to the correct binding point
        if (!m_isBound[binding.bindingIndex] || m_boundBuffers[binding.bindingIndex] != binding.glBuffer)
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding.bindingIndex, binding.glBuffer);
            m_isBound[binding.bindingIndex] = true;
            m_boundBuffers[binding.bindingIndex] = binding.glBuffer;
        }
    }
}
