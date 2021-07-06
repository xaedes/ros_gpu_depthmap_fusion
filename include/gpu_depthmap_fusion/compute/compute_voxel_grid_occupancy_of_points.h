#pragma once
#include <vector>
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeVoxelGridOccupancyOfPoints : public ComputeProgram
{
public:
    virtual ~ComputeVoxelGridOccupancyOfPoints() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_points, 
        uint bufbind_out_occupancy)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "voxel_grid_occupancy_of_points.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_IN_COORDS",  std::to_string(bufbind_points)},
            {"BUFBIND_OUT_OCCUPANCY", std::to_string(bufbind_out_occupancy)}
        });
        num_items.init(m_glProgram, "num_items");
        in_coord_offset.init(m_glProgram, "in_coord_offset");
        out_occupancy_offset.init(m_glProgram, "out_occupancy_offset");
        occupied_value.init(m_glProgram, "occupied_value");
    }
    ProgramUniform<uint> num_items;
    ProgramUniform<uint> in_coord_offset;
    ProgramUniform<uint> out_occupancy_offset;
    ProgramUniform<uint> occupied_value;

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

