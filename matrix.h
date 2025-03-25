#ifndef MATRIX_H
#define MATRIX_H

template<int TRows, int TCols>
class Matrix {
    public:
    double data[TRows][TCols];
    const int cols = TCols;
    const int rows = TRows;

    __device__ __host__ double& at(int row, int col) {
        return data[row][col];
    }

    __device__ __host__ const double& at(int row, int col) const {
        return data[row][col];
    }

    void setData(const double newData[TRows][TCols]) {
        for (int i = 0; i < TRows; i++) {
            for (int j = 0; j < TCols; j++) {
                data[i][j] = newData[i][j];
            }
        }
    }
    template<int TColsB>
    __device__ __host__ void mult(Matrix<TCols, TColsB>* b, Matrix<TRows, TColsB>* result) {

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