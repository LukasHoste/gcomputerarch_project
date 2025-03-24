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
    void set(int row, int col, float value);
    float get(int row, int col);
    Matrix add(Matrix other);
    Matrix mult(Matrix other);
    void print();
    static Matrix fromVect(const std::vector<std::vector<float>>& init);
    void toGpu(Matrix* gpu_matrix);
    void toCpu(Matrix* cpu_matrix);
private:
    int rows;
    int cols;
    bool is_gpu;
    float *data;
};
#endif // MATRIX_H