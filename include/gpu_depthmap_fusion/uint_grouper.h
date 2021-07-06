#pragma once
#include <vector>

// Group a list of integers by its values in O(N) time and O(N+M) memory.
//
// Memory usage is linear to number of items plus maximum integer value / aka
// group number. The resulting item order is stored in groupedIndices, which
// contains indices into the given list of group numbers. The position of the
// groups inside this array is stored in groupStarts and groupSizes.
// groupPointers is only internally used and you can ignore it.
template <typename TUint> // for different number of bits
class UIntGrouper
{
public:
    UIntGrouper()
    {
    }
    void group(
        const std::vector<TUint>& groupNumbers
    )
    {
        uint numItems = groupNumbers.size();
        if (numItems == 0) 
        {
            group(groupNumbers, numItems, 0);
            return;
        }
        uint numGroups = groupNumbers[0];
        for (uint i = 1; i < numItems; ++i)
            if (groupNumbers[i] > numGroups) 
                numGroups = groupNumbers[i];
        numGroups += 1;
        group(groupNumbers, numItems, numGroups);

    }
    void group(
        const std::vector<TUint>& groupNumbers,
        uint numGroups
    )
    {
        uint numItems = groupNumbers.size();
        group(groupNumbers, numItems, numGroups);
    }
    void group(
        const std::vector<uint>& groupNumbers,
        uint numItems,
        uint numGroups
    )
    {
        // std::cout << "IntGrouper::group BEGIN"  << std::endl;

        // std::cout << "groupNumbers " << numItems  << std::endl;
        // for (int k=0; k<numItems; ++k)
        //     std::cout << " " << groupNumbers[k];
        // std::cout << std::endl;
        // std::cout << "numItems " << numItems  << std::endl;
        // std::cout << "numGroups " << numGroups  << std::endl;

        // initialize group info
        groupStarts.resize(numGroups);
        groupSizes.resize(numGroups);
        groupPointers.resize(numGroups);
        
        // when number of groups is not big, we can zero them all another
        // implementation strategy would be to only zero relevant groups. to
        // ensure consistency with remaining group data, each group is assigned
        // a sequence number, which can be used to check whether a group is
        // fresh or old and should therefore be treated as zero
        for (uint k=0; k<numGroups; ++k)
        {
            groupStarts[k] = 0;
            groupSizes[k] = 0;
            groupPointers[k] = 0;
        }
        // determine group sizes
        for (uint i=0; i<numItems; ++i)
        {
            TUint groupNumber = groupNumbers[i];
            ++groupSizes[groupNumber];
        }
        numUniqueGroups = 0;
        // determine group starts
        uint nextGroupStart = 0;
        for (uint k=0; k<numGroups; ++k)
        {
            groupStarts[k] = nextGroupStart;
            nextGroupStart += groupSizes[k];
            if (groupSizes[k] > 0)
                ++numUniqueGroups;
        }
        assert(nextGroupStart == numItems);
        // group items
        groupedIndices.resize(numItems);
        for (uint i = 0; i < numItems; ++i)
        {
            TUint groupNumber = groupNumbers[i];
            uint insertIndex = groupStarts[groupNumber] + groupPointers[groupNumber];
            groupedIndices[insertIndex] = i;
            ++groupPointers[groupNumber];
        }
        // std::cout << "IntGrouper::group END"  << std::endl;
    }

    std::vector<uint> groupedIndices;
    std::vector<uint> groupStarts;
    std::vector<uint> groupSizes;
    std::vector<uint> groupPointers;
    uint numUniqueGroups;
};

