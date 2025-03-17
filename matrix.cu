#include "matrix.h"
#include <stdio.h>

Matrix::Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = (float*) malloc(rows * cols * sizeof(float));

    for (int i = 0; i < rows * cols; i++) {
        this->data[i] = 0;
    }
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
    free(this->data);
    this->data = NULL;
}

Matrix::Matrix(const Matrix& other) {
    this->rows = other.rows;
    this->cols = other.cols;
    this->data = (float*) malloc(this->rows * this->cols * sizeof(float));

    for (int i = 0; i < this->rows * this->cols; i++) {
        this->data[i] = other.data[i];
    }
}

Matrix& Matrix::operator=(const Matrix& other) {
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

void Matrix::print() {
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            printf("%f ", this->data[i * this->cols + j]);
        }
        printf("\n");
    }
}

float Matrix::get(int row, int col) {
    return this->data[row * this->cols + col];
}

void Matrix::set(int row, int col, float value) {
    this->data[row * this->cols + col] = value;
}

Matrix Matrix::add(Matrix other) {
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