#ifndef MATRIX_H
#define MATRIX_H

template<size_t TRows, size_t TCols>
class Matrix {
    public:
    double data[TRows][TCols];
    const size_t cols = TCols;
    const size_t rows = TRows;

    __device__ __host__ double& at(size_t row, size_t col) {
        return data[row][col];
    }

    __device__ __host__ const double& at(size_t row, size_t col) const {
        return data[row][col];
    }

    void setData(const double newData[TRows][TCols]) {
        for (int i = 0; i < TRows; i++) {
            for (int j = 0; j < TCols; j++) {
                data[i][j] = newData[i][j];
            }
        }
    }
    template<size_t TColsB>
    __device__ __host__ void mult(const Matrix<TCols, TColsB>* b, Matrix<TRows, TColsB>* result) const {

        for (int i = 0; i < TRows; i++) {
            for (int j = 0; j < TColsB; j++) {
                double sum = 0;
                for (int k = 0; k < cols; k++) {
                    sum += this->at(i, k) * b->at(k, j);
                }
                result->at(i, j) = sum;
            }
        }
    }


    template<size_t TColsB>
    __device__ __host__ Matrix<TRows, TColsB> operator*(const Matrix<TCols, TColsB>& b) const {
        Matrix<TRows, TColsB> result;
        this->mult(&b, &result);
        return result;
    }

    __device__ __host__ Matrix<TRows, TCols>& operator=(const Matrix<TRows, TCols>& other) {
        for (int i = 0; i < TRows; i++) {
            for (int j = 0; j < TCols; j++) {
                data[i][j] = other.data[i][j];
            }
        }
        return *this;
    }


};

#endif // MATRIX_H