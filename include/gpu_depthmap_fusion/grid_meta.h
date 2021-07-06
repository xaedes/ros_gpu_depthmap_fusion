#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm> 

using cv::Vec;
using cv::norm;

/**
 * @brief      This class describes a grid by its bounds in the world and
 *             cell size.
 *
 *             It provides functions to convert coordinates between world
 *             coordinates, grid cell coordinates and one dimensional cell
 *             indices.
 */
// template<typename TValueType, unsigned int TNumDimensions, typename TCellIndex>
template<typename TValueVec, typename TIntVec, typename TBoolVec, unsigned int TNumDimensions, typename TCellIndex>
class GridMeta
{
public:
    // typedef Vec<TValueType, TNumDimensions> ValueVec;
    // typedef Vec<int, TNumDimensions> IntVec;
    // typedef Vec<bool, TNumDimensions> BoolVec;
    GridMeta()
    {}
    GridMeta(TValueVec lowerBound, TValueVec upperBound, TValueVec cellSize, TBoolVec wrap)
    {
        set(lowerBound, upperBound, cellSize, wrap);
    }

    void set(TValueVec lowerBound, TValueVec upperBound, TValueVec cellSize, TBoolVec wrap)
    {
        this->m_lowerBound = lowerBound;
        this->m_upperBound = upperBound;
        this->m_cellSize = cellSize;
        this->wrap = wrap;
        update();
    }

    void setCellSize(TValueVec cellSize)
    {
        this->m_cellSize = cellSize;
    }

    inline TIntVec gridCoord(TCellIndex cellIndex) const
    {
        TIntVec result;
        result[0] = (int)(cellIndex);
        for (int i = 0; i < TNumDimensions; ++i)
        {
            result[i] = (cellIndex / m_steps[i]) % m_gridSize[i];
            cellIndex -= result[i] * m_steps[i];
        }
        assert(cellIndex == 0);
        return result;
    }

    inline TIntVec gridCoord(TValueVec worldCoord) const
    {
        // auto scaled_coord = (worldCoord - m_lowerBound) / m_cellSize;
        TIntVec result;
        for (int i = 0; i < TNumDimensions; ++i)
        {

            result[i] = (int)((worldCoord[i] - m_lowerBound[i]) / m_cellSize[i]);
            result[i] = result[i] % m_gridSize[i];
            // % can give negative result (up to -(m_gridSize[i]-1))
            if (result[i] < 0) result[i] += m_gridSize[i];
        }
        return result;
    }

    inline TCellIndex cellIndex(TValueVec worldCoord) const
    {

        return cellIndex(gridCoord(worldCoord));
    }

    inline TCellIndex cellIndex(TIntVec gridCoord) const
    {
        TCellIndex index = 0;
        for (int i = 0; i < TNumDimensions; ++i)
        {
            index += m_steps[i] * gridCoord[i];
        }
        return index;
    }

    inline TValueVec worldCoord(TCellIndex cellIndex) const
    {

        return worldCoord(gridCoord(cellIndex));
    }
    inline TValueVec worldCoord(TIntVec gridCoord) const
    {
        TValueVec result;
        for (int i = 0; i < TNumDimensions; ++i)
            result[i] = gridCoord[i] * m_cellSize[i] + m_lowerBound[i];
        return result;
    }

    inline TValueVec lowerBound() const { return m_lowerBound; }
    inline TValueVec upperBound() const { return m_upperBound; }
    inline TValueVec cellSize() const { return m_cellSize; }
    inline TValueVec gridDimensions() const { return m_gridDimensions; }
    inline TIntVec gridSize() const { return m_gridSize; }
    inline TIntVec steps() const { return m_steps; }
    inline TCellIndex numCells() const { return m_numCells; }

    TBoolVec wrap;

    bool bounds_check(int dimension, int& index) const
    {
        if (wrap[dimension])
        {
            index = index % m_gridSize[dimension];
            if (index < 0) index += m_gridSize[dimension];
            return true;
        }
        else
        {
            if (index < 0) return false;
            if (index >= m_gridSize[dimension]) return false;
            return true;
        }
    }

    bool bounds_check(TIntVec& gridCoord) const
    {
        bool inBounds = true;
        for (int i = 0; i < TNumDimensions; ++i)
        {
            inBounds &= bounds_check(i, gridCoord[i]);
            if (!inBounds) break;
        }
        return inBounds;
    }

protected:
    void update()
    {
        for (int i = 0; i < TNumDimensions; ++i)
        {
            float lower = std::min(m_lowerBound[i], m_upperBound[i]);
            float upper = std::max(m_lowerBound[i], m_upperBound[i]);
            m_lowerBound[i] = lower;
            m_upperBound[i] = upper;
        }
        m_gridDimensions = m_upperBound - m_lowerBound;
        m_numCells = 1;
        for (int i = 0; i < TNumDimensions; ++i)
        {
            m_gridSize[i] = (int)ceil(m_gridDimensions[i] / m_cellSize[i]);
            if (m_gridSize[i] < 1) m_gridSize[i] = 1;
            m_steps[i] = m_numCells; // stride for dimension 0 is always 1
            m_numCells *= m_gridSize[i];
        }
    }


    TValueVec m_lowerBound;
    TValueVec m_upperBound;
    TValueVec m_cellSize;

    TValueVec m_gridDimensions;
    TIntVec m_gridSize;
    TIntVec m_steps; // steps between elements in kth dimension (m_steps[0] == 1)
    TCellIndex m_numCells;
};
