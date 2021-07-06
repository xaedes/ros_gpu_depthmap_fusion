#pragma once

template<typename T, uint TNum>
void rotate_vector(const T* mat, const T* vec_in, T* vec_out)
{
    for (int i = 0; i < TNum; ++i)
    {
        vec_out[i] = 0;
        for (int j = 0; j < TNum; ++j)
        {
            vec_out[i] += mat[i*TNum + j] * vec_in[j];
        }
    }
}

template<typename T, uint TNum>
void rotate_vector_inv(const T* mat, const T* vec_in, T* vec_out)
{
    for (int i = 0; i < TNum; ++i)
    {
        vec_out[i] = 0;
        for (int j = 0; j < TNum; ++j)
        {
            // note the indices in mat: its using the transpose of mat
            vec_out[i] += mat[j*TNum + i] * vec_in[j];
        }
    }
}
