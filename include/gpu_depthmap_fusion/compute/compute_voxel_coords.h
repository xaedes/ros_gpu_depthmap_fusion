#pragma once
#include <vector>
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeVoxelCoords : public ComputeProgram
{
public:
    virtual ~ComputeVoxelCoords() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_points, 
        uint bufbind_out_coords)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "compute_voxel_coords.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_POINTS",  std::to_string(bufbind_points)},
            {"BUFBIND_OUT_VOXEL_COORDS", std::to_string(bufbind_out_coords)}
        });
        num_items.init(m_glProgram, "num_items");
        in_point_offset.init(m_glProgram, "in_point_offset");
        out_coord_offset.init(m_glProgram, "out_coord_offset");
        lower_bound.init(m_glProgram, "lower_bound");
        upper_bound.init(m_glProgram, "upper_bound");
        cell_size.init(m_glProgram, "cell_size");
        grid_size.init(m_glProgram, "grid_size");
    }
    ProgramUniform<uint> num_items;
    ProgramUniform<uint> in_point_offset;
    ProgramUniform<uint> out_coord_offset;
    ProgramUniform<glm::vec3> lower_bound;
    ProgramUniform<glm::vec3> upper_bound;
    ProgramUniform<glm::vec3> cell_size;
    ProgramUniform<glm::uvec3> grid_size;

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

