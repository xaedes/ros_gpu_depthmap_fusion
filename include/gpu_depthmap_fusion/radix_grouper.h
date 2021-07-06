#pragma once
#include "radix_sort.h"


template <typename T, typename TUintVec4>
class RadixGrouper
{
public:
    RadixGrouper()
        : sorter(1024)
    {}
    RadixGrouper(int radix_group_size)
        : sorter(radix_group_size)
    {}

    std::vector<uint> groupStarts;
    std::vector<uint> groupSizes;
    std::vector<T> groupValues;

    RadixSorter<T,TUintVec4> sorter;

    void group(const T* valuesPtr, int num)
    {
        sorter.sort(valuesPtr, num);
        makeGroups();
    }
    void group(const std::vector<T>& values)
    {
        sorter.sort(values);
        makeGroups();
    }

protected:

    void makeGroups()
    {
        groupStarts.resize(0);
        groupSizes.resize(0);
        groupValues.resize(0);
        int num = sorter.sortedValues.size();
        if (num == 0)
        {
            return;
        }
        T lastValue = sorter.sortedValues[0];
        uint lastGroupStart = 0;
        for(uint i = 1; i < num; ++i)
        {
            T value = sorter.sortedValues[i];
            if (value != lastValue)
            {
                // start new group, write down last group that is now complete
                groupStarts.push_back(lastGroupStart);
                groupSizes.push_back(i - lastGroupStart);
                groupValues.push_back(lastValue);
                lastGroupStart = i;
                lastValue = value;
            }
        }
        // end of last group, write down last group 
        groupStarts.push_back(lastGroupStart);
        groupSizes.push_back(num - lastGroupStart);
        groupValues.push_back(lastValue);
    }


};
