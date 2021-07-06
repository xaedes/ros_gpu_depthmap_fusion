#version 430 core

// layers_connections

#define GROUPSIZE_X %%GROUPSIZE_X%%
#define GROUPSIZE_Y %%GROUPSIZE_Y%%
#define GROUPSIZE_Z %%GROUPSIZE_Z%%
#define GROUPSIZE (GROUPSIZE_X*GROUPSIZE_Y*GROUPSIZE_Z)

layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

// #define BUFBIND_COORDS                    %%BUFBIND_COORDS%%
#define BUFBIND_LABELS                    %%BUFBIND_LABELS%%
#define BUFBIND_NUM_LABELS_PER_LAYER      %%BUFBIND_NUM_LABELS_PER_LAYER%%
#define BUFBIND_LAYERS_CONNECTIONS_STARTS %%BUFBIND_LAYERS_CONNECTIONS_STARTS%%
#define BUFBIND_LAYERS_CONNECTIONS        %%BUFBIND_LAYERS_CONNECTIONS%%

layout (std430, binding = BUFBIND_LABELS) buffer buf_labels
{
    uint labels[];
};

layout (std430, binding = BUFBIND_NUM_LABELS_PER_LAYER) buffer buf_numLabelsPerLayer
{
    // contains num_layers items
    uint numLabelsPerLayer[];
};

layout (std430, binding = BUFBIND_LAYERS_CONNECTIONS_STARTS) buffer buf_connStarts
{
    // contains num_layers-1 items
    uint connStarts[];
};

layout (std430, binding = BUFBIND_LAYERS_CONNECTIONS) buffer buf_outConnections
{
    // contains mask values for connection matrices as uints
    uint outConnections[];
};

uniform uint num_items = 0;
uniform uint num_layers;
uniform uint layer_size;
uniform uint width;

const int neighbors_size_x = 0;
const int neighbors_size_y = 0;
// uint height;

uvec2 get_pixel(uint linear_idx)
{
    uint x = uint(mod(linear_idx, width));
    uint y = uint(linear_idx / width);
    return uvec2(x, y);
}

uint get_pixel_idx(uint x, uint y)
{
    return x + y*width;
}

uint readLabel(uint layer, uint point_idx)
{
    uint label_idx = layer * layer_size + point_idx;
    uint label = labels[label_idx];
    return label;
}


void addConnection(uint lowerLayer, uint labelA, uint labelB)
{
    uint rowStep = numLabelsPerLayer[lowerLayer+1];
    uint offset = connStarts[lowerLayer];

    uint connection_idx = offset + labelA * rowStep + labelB;
    outConnections[connection_idx] = 1;
}

void add_between_layers(uint lowerLayer, uvec2 pixel, uint dx, uint dy)
{
    uint nx = pixel.x + dx;
    uint ny = pixel.y + dy;
    if (lowerLayer+1 < num_layers && 0 <= nx && nx < width && 0 <= ny && ny < layer_size / width)
    {
        uint point_idx = get_pixel_idx(pixel.x, pixel.y);
        uint point_idx2 = get_pixel_idx(nx, ny);
        uint labelA = readLabel(lowerLayer, point_idx);
        uint labelB = readLabel(lowerLayer+1, point_idx2);
        addConnection(lowerLayer, labelA, labelB);
    }
}

// is called for each on voxel grid (except first layer)

// lookup labels around point and set respective connection entries
void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;
    // height = layer_size / width;
    // uint coord = coords[global_idx];

    uint layerA = global_idx / layer_size;
    
    // if (layerB >= num_layers) return; // debug

    uint point_idx = global_idx - layerA * layer_size;
    uvec2 pixel = get_pixel(point_idx);

    add_between_layers(layerA, pixel, 0, 0);
    if (neighbors_size_x + neighbors_size_y > 0)
    for (int dx=-neighbors_size_x;dx<=neighbors_size_x;++dx)
        for (int dy=-neighbors_size_y;dy<=neighbors_size_y;++dy)
            add_between_layers(layerA, pixel, dx, dy);



}
