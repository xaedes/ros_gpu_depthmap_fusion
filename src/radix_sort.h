#pragma once
// #include <glm/glm.hpp>
// #include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <cstdint>
#include <omp.h>

typedef uint32_t uint;


template <typename T, typename TUintVec4>
void radixSortPartCountDigitsNth(
    int start, int num, int nthDigit,
    int digitOffset,
    const std::vector<T>& values,
    std::vector<TUintVec4>& digitCounts
)
{
    for (int i = 0; i < num; ++i)
    {
        T val = values[start + i];
        uint8_t digit = (val >> (nthDigit * 8)) & 0xff;
        ++digitCounts[digitOffset + digit][nthDigit];
    }
}
template <typename T, typename TUintVec4>
void radixSortPartCountDigits(
    int start, int num, 
    int digitOffset,
    const std::vector<T>& values,
    std::vector<TUintVec4>& digitCounts
)
{
    for (int i = 0; i < num; ++i)
    {
        T val = values[start + i];
        uint8_t digit0 = val & 0xff;
        uint8_t digit1 = (val >> 8) & 0xff;
        uint8_t digit2 = (val >> 16) & 0xff;
        uint8_t digit3 = (val >> 24) & 0xff;
        ++digitCounts[digitOffset + digit0][0];
        ++digitCounts[digitOffset + digit1][1];
        ++digitCounts[digitOffset + digit2][2];
        ++digitCounts[digitOffset + digit3][3];
    }
}
template <typename T, typename TUintVec4>
void radixSortPartByNthDigit(
    //const std::vector<T>& values,
    int start, int num, int nthDigit,
    int digitOffset,
    const std::vector<TUintVec4>& digitStarts,
    std::vector<TUintVec4>& digitPointers,
    const std::vector<uint>& inIndices,
    const std::vector<T>& inValues,
    std::vector<uint>& sortedIndices,
    std::vector<T>& sortedValues
)
{
    // sort by nth digit 
    for (int i = 0; i < num; ++i)
    {
        uint idx = inIndices[start + i];
        T val = inValues[start + i];
        uint8_t digit = (val >> (nthDigit * 8)) & 0xff;
        uint digitStart = digitStarts[digitOffset + digit][nthDigit];
        uint& digitPointer = digitPointers[digitOffset + digit][nthDigit];
        uint targetIndex = start + digitStart + digitPointer;
        ++digitPointer;
        sortedIndices[targetIndex] = idx;
        sortedValues[targetIndex] = val;
    }
}
template <typename T, typename TUintVec4>
void radixSortRedistribute(
    int numGroups, uint group_size,
    int nthDigit, int digit, 
    const std::vector<TUintVec4>& digitStarts,
    const std::vector<TUintVec4>& digitCounts,
    const std::vector<uint>& inIndices,
    const std::vector<T>& inValues,
    std::vector<uint>& outIndices,
    std::vector<T>& outValues
)
{
    // start of digit block across all groups
    uint start = digitStarts[digit][nthDigit];

    // collect items with digit out of each group
    for (int i = 0; i < numGroups; ++i)
    {
        uint digitOffset = (i + 1) * 256;
        uint digitCountInGroup = digitCounts[digitOffset + digit][nthDigit];
        if (digitCountInGroup == 0) continue;
        uint digitStartInGroup = digitStarts[digitOffset + digit][nthDigit];
        uint groupStart = i * group_size;
        for (int j = 0; j < digitCountInGroup; ++j)
        {
            uint target = start + j;
            outIndices[target] = inIndices[groupStart + digitStartInGroup + j];
            outValues[target] = inValues[groupStart + digitStartInGroup + j];
        }
        start += digitCountInGroup;
    }
}

template <typename T, typename TUintVec4>
void radixWithRedistribution(
    int group_size,
    const std::vector<T>& values,
    //const T* values,
    //int numItems,
    std::vector<uint>& sortedIndices,
    std::vector<T>& sortedValues,
    std::vector<uint>& tmpSortedIndices,
    std::vector<T>& tmpSortedValues,
    std::vector<TUintVec4>& digitCounts,
    std::vector<TUintVec4>& digitStarts,
    std::vector<TUintVec4>& digitPointers
)
{
    int numItems = values.size();
    int numGroups = (int)(ceil((float)numItems / (float)group_size));
    if (numGroups <= 0)
    {
        numGroups = 1;
        group_size = numItems;
    }
    // digit counters for radix sort of each workgroup, and total counts in first block of 256

    digitCounts.resize(256 * (numGroups + 1));
    digitStarts.resize(256 * (numGroups + 1));
    digitPointers.resize(256 * (numGroups + 1));

    #pragma omp parallel for
    for (int k = 0; k < digitPointers.size(); ++k)
    {
        digitCounts[k] = TUintVec4(0, 0, 0, 0);
        digitPointers[k] = TUintVec4(0, 0, 0, 0);
    }
    #pragma omp parallel for
    for (int i = 0; i < numItems; ++i)
    {
        sortedIndices[i] = i;
        sortedValues[i] = values[i];
    }
    // count digit occurences for each group
    #pragma omp parallel for
    for (int i = 0; i < numGroups; ++i)
    {
        int start = group_size * i;
        int end = start + group_size;
        if (end > numItems) end = numItems;
        int num = end - start;
        radixSortPartCountDigits(start, num, (i + 1) * 256, sortedValues, digitCounts);
    }
    // sum digit occurences of each group into first block of 256
    #pragma omp parallel for
    for (int k = 0; k < 256; ++k)
    {
        digitCounts[k] = TUintVec4(0, 0, 0, 0);
        for (int i = 0; i < numGroups; ++i)
        {
            digitCounts[k] += digitCounts[(i + 1) * 256 + k];
        }
    }
    // compute starts of each digit block for each group and for total
    #pragma omp parallel for
    for (int i = -1; i < numGroups; ++i)
        // start from -1, to also include the first block containing total values
    {
        uint digitOffset = (i + 1) * 256;
        digitStarts[digitOffset] = TUintVec4(0, 0, 0, 0);
        for (int k = 1; k < 256; ++k)
        {
            digitStarts[digitOffset + k] = digitStarts[digitOffset + k - 1] + digitCounts[digitOffset + k - 1];
        }
    }
    // sort by first digit in each group

    for (int nthDigit = 0; nthDigit < 4; ++nthDigit)
    {
        // all items have digit==0 , skip sort of this nthDigit, cause there is nothing to do
        if (digitCounts[0][nthDigit] == numItems) continue;
        #pragma omp parallel for
        for (int i = 0; i < numGroups; ++i)
        {
            int start = group_size * i;
            int end = start + group_size;
            if (end > numItems) end = numItems;
            int num = end - start;
            radixSortPartByNthDigit(
                start, num, nthDigit, (i + 1) * 256,
                digitStarts, digitPointers,
                sortedIndices, sortedValues,
                tmpSortedIndices, tmpSortedValues
            );
        }
        // redistribute items across work groups 
        // by collecting items for each digit
        #pragma omp parallel for
        for (int k = 0; k < 256; ++k)
        {
            radixSortRedistribute(
                numGroups, group_size, nthDigit, k,
                digitStarts, digitCounts,
                tmpSortedIndices, tmpSortedValues,
                sortedIndices, sortedValues
            );
        }
        if (nthDigit == 3) break;
        // recount nth digit occurences for each group
        #pragma omp parallel for
        for (int i = 0; i < numGroups; ++i)
        {
            for (int k = 0; k < 256; ++k)
            {
                digitCounts[(i + 1) * 256 + k][nthDigit + 1] = 0;
            }
            int start = group_size * i;
            int end = start + group_size;
            if (end > numItems) end = numItems;
            int num = end - start;
            radixSortPartCountDigitsNth(start, num, nthDigit + 1, (i + 1) * 256, sortedValues, digitCounts);
        }
        // recompute starts of nth digit block for each group 
        #pragma omp parallel for
        for (int i = 0; i < numGroups; ++i)
            // start from 0, to NOT include the first block containing total values
        {
            uint digitOffset = (i + 1) * 256;
            digitStarts[digitOffset][nthDigit + 1] = 0;
            for (int k = 1; k < 256; ++k)
            {
                digitStarts[digitOffset + k][nthDigit + 1] = digitStarts[digitOffset + k - 1][nthDigit + 1] + digitCounts[digitOffset + k - 1][nthDigit + 1];
            }
        }
    }
}

//template <typename T>
//void radixWithRedistribution(
//    int group_size,
//    const std::vector<T>& values,
//    std::vector<uint>& sortedIndices,
//    std::vector<T>& sortedValues,
//    std::vector<uint>& tmpSortedIndices,
//    std::vector<T>& tmpSortedValues,
//    std::vector<TUintVec4>& digitCounts,
//    std::vector<TUintVec4>& digitStarts,
//    std::vector<TUintVec4>& digitPointers
//)
//{
//    radixWithRedistribution(
//        group_size,
//        values.data(),
//        values.size(),
//        sortedIndices,
//        sortedValues,
//        tmpSortedIndices,
//        tmpSortedValues,
//        digitCounts,
//        digitStarts,
//        digitPointers
//    );
//}



template <typename T, typename TUintVec4>
void radixSortPart(
    const std::vector<T>& values,
    int start, int num,
    std::vector<uint>& sorted,
    std::vector<uint>& buf,
    std::vector<T>& valBuf,
    std::vector<T>& valBuf2
)
{
    TUintVec4 digitCounts[256];
    TUintVec4 digitStarts[256];
    TUintVec4 digitPointers[256];
    for (int i = 0; i < 256; ++i)
    {
        digitCounts[i] = TUintVec4(0, 0, 0, 0);
        digitPointers[i] = TUintVec4(0, 0, 0, 0);
    }
    for (int i = 0; i < num; ++i)
    {
        T val = values[start + i];
        uint8_t digit0 = val & 0xff;
        uint8_t digit1 = (val >> 8) & 0xff;
        uint8_t digit2 = (val >> 16) & 0xff;
        uint8_t digit3 = (val >> 24) & 0xff;
        ++digitCounts[digit0].x;
        ++digitCounts[digit1].y;
        ++digitCounts[digit2].z;
        ++digitCounts[digit3].w;
    }
    digitStarts[0] = TUintVec4(0, 0, 0, 0);
    for (int i = 1; i < 256; ++i)
    {
        digitStarts[i] = digitStarts[i - 1] + digitCounts[i - 1];
    }
    // sort by 1st digit (least significant)
    for (int idx = 0; idx < num; ++idx)
    {
        T val = values[start + idx];
        uint8_t digit0 = val & 0xff;
        uint targetIndex = start + digitStarts[digit0].x + digitPointers[digit0].x;
        ++digitPointers[digit0].x;
        buf[targetIndex] = start + idx;
        valBuf2[targetIndex] = val;
    }
    // sort by 2nd digit
    for (int i = 0; i < num; ++i)
    {
        int idx = buf[start + i];
        T val = valBuf2[start + i];
        uint8_t digit1 = (val >> 8) & 0xff;
        uint targetIndex = start + digitStarts[digit1].y + digitPointers[digit1].y;
        ++digitPointers[digit1].y;
        sorted[targetIndex] = idx;
        valBuf[targetIndex] = val;
    }
    // sort by 3rd digit
    for (int i = 0; i < num; ++i)
    {
        int idx = sorted[start + i];
        T val = valBuf[start + i];
        uint8_t digit2 = (val >> 16) & 0xff;
        uint targetIndex = start + digitStarts[digit2].z + digitPointers[digit2].z;
        ++digitPointers[digit2].z;
        buf[targetIndex] = idx;
        valBuf2[targetIndex] = val;
    }
    // sort by 4th digit (most significant)
    for (int i = 0; i < num; ++i)
    {
        int idx = buf[start + i];
        T val = valBuf2[start + i];
        uint8_t digit3 = (val >> 24) & 0xff;
        uint targetIndex = start + digitStarts[digit3].w + digitPointers[digit3].w;
        ++digitPointers[digit3].w;
        sorted[targetIndex] = idx;
        valBuf[targetIndex] = val;
    }
}

template <typename T, typename TUintVec4>
void radixSortInParts(
    int group_size,
    const std::vector<T>& values,
    std::vector<uint>& sorted,
    std::vector<uint>& buf,
    std::vector<T>& valBuf,
    std::vector<T>& valBuf2
)
{
    int numItems = values.size();
    sorted.resize(numItems);
    buf.resize(numItems);
    valBuf.resize(numItems);
    valBuf2.resize(numItems);
    int numGroups = (int)(ceil((float)numItems / (float)group_size));
    if (numGroups <= 0)
    {
        numGroups = 1;
        group_size = numItems;
    }
    // omp_set_num_threads(64);
    //omp_set_num_threads(512);
    #pragma omp parallel for
    for (int i = 0; i < numGroups; ++i)
    {
        int start = group_size * i;
        int end = start + group_size;
        if (end > numItems) end = numItems;
        int num = end - start;
        radixSortPart<T,TUintVec4>(values, start, num, sorted, buf, valBuf, valBuf2);
    }

}

template <typename T, typename TUintVec4>
void radixSort(
    const std::vector<T>& values,
    std::vector<uint>& sorted,
    std::vector<uint>& buf,
    std::vector<T>& valBuf,
    std::vector<T>& valBuf2
)
{
    TUintVec4 digitCounts[256];
    TUintVec4 digitStarts[256];
    TUintVec4 digitPointers[256];
    for (int i = 0; i < 256; ++i)
    {
        digitCounts[i] = TUintVec4(0, 0, 0, 0);
        digitPointers[i] = TUintVec4(0, 0, 0, 0);
    }
    int num = values.size();
    sorted.resize(num);
    buf.resize(num);
    valBuf.resize(num);
    valBuf2.resize(num);
    for (int i = 0; i < num; ++i)
    {
        T val = values[i];
        uint8_t digit0 = val & 0xff;
        uint8_t digit1 = (val >> 8) & 0xff;
        uint8_t digit2 = (val >> 16) & 0xff;
        uint8_t digit3 = (val >> 24) & 0xff;
        ++digitCounts[digit0].x;
        ++digitCounts[digit1].y;
        ++digitCounts[digit2].z;
        ++digitCounts[digit3].w;
    }
    digitStarts[0] = TUintVec4(0, 0, 0, 0);
    for (int i = 1; i < 256; ++i)
    {
        digitStarts[i] = digitStarts[i - 1] + digitCounts[i - 1];
    }
    // sort by 1st digit (least significant)
    for (int idx = 0; idx < num; ++idx)
    {
        T val = values[idx];
        uint8_t digit0 = val & 0xff;
        uint targetIndex = digitStarts[digit0].x + digitPointers[digit0].x;
        ++digitPointers[digit0].x;
        buf[targetIndex] = idx;
        valBuf2[targetIndex] = val;
        //targets[idx] = targetIndex;
        //buf[idx] = targetIndex;
        //valBuf2[idx] = val;
    }
    // sort by 2nd digit
    for (int i = 0; i < num; ++i)
    {
        int idx = buf[i];
        T val = valBuf2[i];
        //T val = values[idx];
        //T val = values[i];
        uint8_t digit1 = (val >> 8) & 0xff;
        uint targetIndex = digitStarts[digit1].y + digitPointers[digit1].y;
        ++digitPointers[digit1].y;
        //sorted[idx] = targetIndex;
        sorted[targetIndex] = idx;
        valBuf[targetIndex] = val;
        //targets[i] = targetIndex;
        //sorted[i] = targetIndex;
        //valBuf[i] = val;
    }
    // sort by 3rd digit
    for (int i = 0; i < num; ++i)
    {
        int idx = sorted[i];
        T val = valBuf[i];
        //T val = values[i];
        //T val = values[idx];
        uint8_t digit2 = (val >> 16) & 0xff;
        uint targetIndex = digitStarts[digit2].z + digitPointers[digit2].z;
        ++digitPointers[digit2].z;
        buf[targetIndex] = idx;
        valBuf2[targetIndex] = val;
        //targets[i] = targetIndex;
        //buf[i] = targetIndex;
        //valBuf2[i] = val;
    }
    // sort by 4th digit (most significant)
    for (int i = 0; i < num; ++i)
    {
        int idx = buf[i];
        T val = valBuf2[i];
        //T val = values[i];
        //T val = values[idx];
        uint8_t digit3 = (val >> 24) & 0xff;
        uint targetIndex = digitStarts[digit3].w + digitPointers[digit3].w;
        ++digitPointers[digit3].w;
        //sorted[targetIndex] = val;
        sorted[targetIndex] = idx;
        valBuf[targetIndex] = val;
        //targets[i] = targetIndex;
        //sorted[i] = targetIndex;
        //valBuf[i] = val;
    }
    //for (int i = 0; i < num; ++i)
        //sorted[i] = values[i];
}