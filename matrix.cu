#include "matrix.h"
#include <stdio.h>

Matrix::Matrix(int rows, int cols) {
    this->is_gpu = false;
    this->rows = rows;
    this->cols = cols;
    this->data = (float*) malloc(rows * cols * sizeof(float));

    for (int i = 0; i < rows * cols; i++) {
        this->data[i] = 0;
    }
}

Matrix::Matrix() {
    this->rows = 0;
    this->cols = 0;
    this->data = NULL;
    this->is_gpu = false;
}

Matrix Matrix::fromVect(const std::vector<std::vector<float>>& init) {
    Matrix result = Matrix(init.size(), init[0].size());

    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            result.data[i * result.cols + j] = init[i][j];
        }
    }
    return result;
}

 Matrix::~Matrix() {
    if (this->is_gpu) {
        return;
    }
    free(this->data);
    this->data = NULL;
}

Matrix::Matrix(const Matrix& other) {
    if (this->is_gpu || other.is_gpu) {
        return;
    }
    this->rows = other.rows;
    this->cols = other.cols;
    this->data = (float*) malloc(this->rows * this->cols * sizeof(float));

    for (int i = 0; i < this->rows * this->cols; i++) {
        this->data[i] = other.data[i];
    }
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this->is_gpu || other.is_gpu) {
        return *this;
    }
    if (this->data != NULL) {
        free(this->data);
    }
    this->rows = other.rows;
    this->cols = other.cols;
    this->data = (float*) malloc(this->rows * this->cols * sizeof(float));

    for (int i = 0; i < this->rows * this->cols; i++) {
        this->data[i] = other.data[i];
    }
    return *this;
}

__device__  __host__ void Matrix::print() {
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            printf("%f ", this->data[i * this->cols + j]);
        }
        printf("\n");
    }
}

__device__  __host__  float Matrix::get(int row, int col) {
    return this->data[row * this->cols + col];
}

__device__  __host__ void Matrix::set(int row, int col, float value) {
    this->data[row * this->cols + col] = value;
}

__device__  __host__  Matrix Matrix::add(Matrix other) {
    Matrix result = Matrix(this->rows, this->cols);
    if (this->rows != other.rows || this->cols != other.cols) {
        printf("Matrix dimensions must match\n");
        return result;
    }
    for(int i = 0; i < this->rows * this->cols; i++) {
        result.data[i] = this->data[i] + other.data[i];
    }
    return result;
}

Matrix Matrix::mult(Matrix other) {
    Matrix result = Matrix(this->rows, other.cols);
    if (this->cols != other.rows) {
        printf("Matrix dimensions must match\n");
        return result;
    }
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            float sum = 0;
            for (int k = 0; k < this->cols; k++) {
                sum += this->data[i * this->cols + k] * other.data[k * other.cols + j];
            }
            result.data[i * result.cols + j] = sum;
        }
    }
    return result;
}


// already allocated!
void Matrix::toGpu(Matrix* gpu_matrix) {
    Matrix copy = Matrix();
    copy.rows = this->rows;
    copy.cols = this->cols;
    copy.is_gpu = true;
    // copy this data to gpu and get ptr
    float* memPtr;
    cudaMalloc(&memPtr, this->rows * this->cols * sizeof(float));
    cudaMemcpy(memPtr, this->data, this->rows * this->cols * sizeof(float), cudaMemcpyHostToDevice);
    copy.data = memPtr;
    cudaMemcpy(gpu_matrix, &copy, sizeof(Matrix), cudaMemcpyHostToDevice);
}


void Matrix::toCpu(Matrix* cpu_matrix) {
    cudaMemcpy(cpu_matrix, this, sizeof(Matrix), cudaMemcpyDeviceToHost);
    float* data = (float*) malloc(this->rows * this->cols * sizeof(float));
    cpu_matrix->is_gpu = false;
    cudaMemcpy(data, cpu_matrix->data, this->rows * this->cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(cpu_matrix->data);
    cpu_matrix->data = data;
}

__device__ __host__ void multAndStoreInB(Matrix* a, Matrix* b, float* tempBuffer) {
    // b gets copied into the temp buffer
    /*for (int i = 0; i < b->rows; i++) {
        for (int j = 0; j < b->cols; i++) {
            tempBuffer[i *b->cols + j] = b->data[i * b->cols + j];
        }
    }*/


    if (a->cols != b->rows) {
        printf("Matrix dimensions must match\n");
        return;
    }
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            tempBuffer[i * b->cols + j] = sum;
        }
    }

    for (int i = 0; i < b->rows; i++) {
        for (int j = 0; j < b->cols; i++) {
            b->data[i * b->cols + j] =  tempBuffer[i *b->cols + j];
        }
    }
}