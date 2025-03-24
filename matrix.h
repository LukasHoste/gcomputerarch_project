#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
class Matrix {
public:
Matrix(int rows, int cols);
Matrix(const Matrix& other);
Matrix();
 Matrix& operator=(const Matrix& other);
~Matrix();
__device__ __host__ void set(int row, int col, float value);
__device__ __host__ float get(int row, int col);
Matrix add(Matrix other);
Matrix mult(Matrix other);
__device__ __host__ void print();
__device__ __host__ static Matrix fromVect(const std::vector<std::vector<float>>& init);
__device__ __host__ void toGpu(Matrix* gpu_matrix);
__device__ __host__ void toCpu(Matrix* cpu_matrix);
    int rows;
    int cols;
    bool is_gpu;
    float *data;
};


__device__ __host__  void multAndStoreInB(Matrix* a, Matrix* b, float* tempBuffer);
#endif // MATRIX_H