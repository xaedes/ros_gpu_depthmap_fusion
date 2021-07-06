#pragma once

#include "radix_grouper.h"
#include "grid_meta.h"

#include <opencv2/opencv.hpp>
#include <vector>

template <typename TValueVec, unsigned int TNumDimensions, typename TCellIndex>
int averageGridCells(
    RadixGrouper<TCellIndex, cv::Vec<uint,4>> radixGrouper,
    const std::vector<TValueVec>& points,
    std::vector<TValueVec>& out_points,
    bool enableParallel
)
{
    out_points.resize(radixGrouper.groupStarts.size());
    #pragma omp parallel for if(enableParallel)
    for (int i=0; i<radixGrouper.groupStarts.size(); ++i)
    {
        auto groupStart = radixGrouper.groupStarts[i];
        auto groupSize = radixGrouper.groupSizes[i];

        assert(groupSize > 0);

        TValueVec sum;
        for(int j=0; j<TNumDimensions; ++j) sum[j] = 0;
        uint count = 0;
        for (int k=0; k<groupSize; ++k)
        {
            uint idx = radixGrouper.sorter.sortedIndices[groupStart + k];
            const auto& val = points[idx];
            sum += val;
            ++count;
        }
        
        assert(count > 0);

        auto mean = sum;

        for(int j=0; j<TNumDimensions; ++j) 
        {
            mean[j] /= count;
        }
        out_points[i] = mean;
    }
    return out_points.size();
}

template <typename TValueVec, typename TGridMeta, unsigned int TNumDimensions, typename TCellIndex>
int occupiedGridCells(
    RadixGrouper<TCellIndex, cv::Vec<uint,4>> radixGrouper,
    TGridMeta gridMeta,
    std::vector<TValueVec>& out_points,
    bool enableParallel
)
{
    out_points.resize(radixGrouper.groupStarts.size());
    // #pragma omp parallel for if(enableParallel)
    for (int i=0; i<radixGrouper.groupStarts.size(); ++i)
    {
        auto groupStart = radixGrouper.groupStarts[i];
        auto groupSize = radixGrouper.groupSizes[i];
        TCellIndex groupValue = radixGrouper.groupValues[i];

        assert(groupSize > 0);

        out_points[i] = gridMeta.worldCoord(groupValue);
    }
    return out_points.size();
}


template <typename TValueVec, typename TGridMeta, unsigned int TNumDimensions, typename TCellIndex>
int voxelize(
    RadixGrouper<TCellIndex, cv::Vec<uint,4>> radixGrouper,
    TGridMeta gridMeta,
    const TCellIndex* voxelCoords,
    int num,
    const std::vector<TValueVec>& points,
    std::vector<TValueVec>& out_points, 
    bool averageVoxels,
    bool enableParallel
)
{
    radixGrouper.group(voxelCoords, num);
    if (averageVoxels)
    {
        return averageGridCells<TValueVec, TNumDimensions, TCellIndex>(
            radixGrouper,
            points,
            out_points,
            enableParallel
        );
    }
    else
    {
        return occupiedGridCells<TValueVec, TGridMeta, TNumDimensions, TCellIndex>(
            radixGrouper,
            gridMeta,
            out_points,
            enableParallel
        );
    }
}
