#pragma once
#include <vector>
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeLayersConnections : public ComputeProgram
{
public:
    virtual ~ComputeLayersConnections() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_labels, 
        uint bufbind_num_labels_per_layer, 
        uint bufbind_layers_connections_starts, 
        uint bufbind_layers_connections)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "layers_connections.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_LABELS",                    std::to_string(bufbind_labels)},
            {"BUFBIND_NUM_LABELS_PER_LAYER",      std::to_string(bufbind_num_labels_per_layer)},
            {"BUFBIND_LAYERS_CONNECTIONS_STARTS", std::to_string(bufbind_layers_connections_starts)},
            {"BUFBIND_LAYERS_CONNECTIONS",        std::to_string(bufbind_layers_connections)}
        });
        num_items.init(m_glProgram, "num_items");
        num_layers.init(m_glProgram, "num_layers");
        layer_size.init(m_glProgram, "layer_size");
        width.init(m_glProgram, "width");
    }

    ProgramUniform<uint> num_items;
    ProgramUniform<uint> num_layers;
    ProgramUniform<uint> layer_size;
    ProgramUniform<uint> width;

    virtual ComputeProgram& dispatch()
    {
        dispatch(num_items.get(), 1, 1);
        return *this;
    }
    virtual ComputeProgram& dispatch(int count)
    {
        dispatch(count, 1, 1);
        return *this;
    }
    virtual ComputeProgram& dispatch(int x, int y, int z)
    {
        ComputeProgram::dispatch(x,y,z,m_groupsize_x,m_groupsize_y,m_groupsize_z);
        return *this;
    }
protected:
    uint m_groupsize_x;
    uint m_groupsize_y;
    uint m_groupsize_z;
};

