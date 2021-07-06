#pragma once
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeConvertDepthmapToPoints : public ComputeProgram
{
public:
    virtual ~ComputeConvertDepthmapToPoints() {}
    virtual void init(const std::string& shader_path,
        uint bufbind_depth_pairs, 
        uint bufbind_mask, 
        uint bufbind_out_pointsa, 
        uint bufbind_out_pointsb, 
        uint bufbind_out_pointsc, 
        uint bufbind_rectify_map)

    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "convert_depthmap_to_points.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_DEPTH_PAIRS", std::to_string(bufbind_depth_pairs)},
            {"BUFBIND_MASK",        std::to_string(bufbind_mask)},
            {"BUFBIND_OUT_POINTSA",  std::to_string(bufbind_out_pointsa)},
            {"BUFBIND_OUT_POINTSB",  std::to_string(bufbind_out_pointsb)},
            {"BUFBIND_OUT_POINTSC",  std::to_string(bufbind_out_pointsc)},
            {"BUFBIND_RECTIFY_MAP", std::to_string(bufbind_rectify_map)}
        });
        depth_scale.init(m_glProgram, "depth_scale");
        fx.init(m_glProgram, "fx");
        fy.init(m_glProgram, "fy");
        cx.init(m_glProgram, "cx");
        cy.init(m_glProgram, "cy");
        width.init(m_glProgram, "width");
        height.init(m_glProgram, "height");
        num_items.init(m_glProgram, "num_items");
        depth_pair_offset.init(m_glProgram, "depth_pair_offset");
        point_offset.init(m_glProgram, "point_offset");
        mask_offset.init(m_glProgram, "mask_offset");
        rectify_offset.init(m_glProgram, "rectify_offset");
        transform_world.init(m_glProgram, "transform_world");
        transform_crop.init(m_glProgram, "transform_crop");

    }
    ProgramUniform<float> depth_scale;
    ProgramUniform<float> fx;
    ProgramUniform<float> fy;
    ProgramUniform<float> cx;
    ProgramUniform<float> cy;
    ProgramUniform<uint> width;
    ProgramUniform<uint> height;
    ProgramUniform<uint> num_items;
    ProgramUniform<uint> depth_pair_offset;
    ProgramUniform<uint> point_offset;
    ProgramUniform<uint> mask_offset;
    ProgramUniform<uint> rectify_offset;
    ProgramUniform<cv::Matx44f> transform_world;
    ProgramUniform<cv::Matx44f> transform_crop;
    
    virtual ComputeProgram& dispatch()
    {
        ComputeConvertDepthmapToPoints::dispatch(width.get()*height.get(), 1, 1);
        // ComputeConvertDepthmapToPoints::dispatch(width.get(), height.get(), 1);
        // dispatch(num_items.get(), 1, 1);
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