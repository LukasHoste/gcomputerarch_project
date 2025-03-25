#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include "matrix.h"

void save_image_array(uint8_t* image_array, int width, int height, int channels) {
    /*
     * Save the data of an (RGB) image as a pixel map.
     * 
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     * 
     */            
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen("./output_image.ppm","wb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    
    // Configure the file
    fprintf(imageFile,"P6\n");               // P6 filetype
    fprintf(imageFile,"%d %d\n", width, height);      // dimensions
    fprintf(imageFile,"255\n");              // Max pixel
    
    // Write the image
    fwrite(image_array, 1, width * height * channels, imageFile);
    
    // Close the file
    fclose(imageFile);
}


void save_black_white_image(uint8_t* image_array, int width, int height) {
    uint8_t* color_image = (uint8_t*)malloc(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        color_image[i * 3] = image_array[i];
        color_image[i * 3 + 1] = image_array[i];
        color_image[i * 3 + 2] = image_array[i];
    }
    save_image_array(color_image, width, height, 3);
    free(color_image);
}


void setBottomLeftMatrix(Matrix<3,3>* matrix) {
    const double newData[3][3] = {
        {0.5, 0, 0},
        {0, 0.5, 0},
        {0, 0, 1},
    };

    matrix->setData(newData);
}

void setBottomRightMatrix(Matrix<3,3>* matrix) {
    const double newData[3][3] = {
        {0.5, 0, 0.5},
        {0, 0.5, 0},
        {0, 0, 1},
    };

    matrix->setData(newData);
}

void setTopMatrix(Matrix<3,3>* matrix) {
    const double newData[3][3] = {
        {0.5, 0, 0.25},
        {0, 0.5, 0.5},
        {0, 0, 1},
    };

    matrix->setData(newData);
}


Matrix<3,3>* get_random_trig_point(Matrix<3,3>* bottomLeftMatrix, Matrix<3,3>* bottomRightMatrix, Matrix<3,3>* topMatrix) {
    int random = rand() % 3;
    if (random < 1) {
        return bottomLeftMatrix;
    } else if (random < 2) {
        return bottomRightMatrix;
    } else {
        return topMatrix;
    }
}



void create_triangle(Matrix<3, 1>* points, int amount, int iterations,Matrix<3, 1>* buffer ,Matrix<3,3>* bottomLeftMatrix, Matrix<3,3>* bottomRightMatrix, Matrix<3,3>* topMatrix) {
    // the tirnalgle is on a one by one grid
    for (int j = 0; j < iterations; j++) {
        for (int i = 0; i < amount; i++) {
            Matrix<3, 3>* random_trig_point = get_random_trig_point(bottomLeftMatrix, bottomRightMatrix, topMatrix);
            Matrix<3,1>* current_point = points + i;
            Matrix<3,1>* buffer_point = buffer + i;
            random_trig_point->mult(current_point, buffer_point);
            *current_point = *buffer_point;
        }
    }

}

Matrix<3, 1>* generate_random_points(int amount) {
    Matrix<3, 1>* points = (Matrix<3, 1>*)malloc(amount * sizeof(Matrix<3, 1>));
    for (int i = 0; i < amount; i++) {
        double data[3][1] = {
            {(double)rand() / RAND_MAX},
            {(double)rand() / RAND_MAX},
            {1.0},
        };
        points[i].setData(data);
    }
    return points;
}


uint8_t* scale_to_image(Matrix<3, 1>* points, int amount, int width, int height) {
    uint8_t* image_array = (uint8_t*)calloc(width * height, sizeof(uint8_t));

    for (int i = 0; i < amount; i++) {
        Matrix<3, 1> current_point = points[i];
        int x = fminf(current_point.at(0, 0) * width, width - 1);
        int y = fminf(height - current_point.at(1, 0)* height, height - 1);
        image_array[y * width + x] = 255;
    }
    return image_array;
}



int main() {
    srand(1000);
    int width = 300;
    int height = 300;
    int image_size = width * height;
    
    // Generate random points
    int amount = 100000;
    printf("Generating random points...\n");
    Matrix<3, 1>* points = generate_random_points(amount);
    Matrix<3, 1>* buffer = (Matrix<3, 1>*)malloc(amount * sizeof(Matrix<3, 1>));
    Matrix<3,3> bottomLeftMatrix;
    Matrix<3,3> bottomRightMatrix;
    Matrix<3,3> topMatrix;
    setBottomLeftMatrix(&bottomLeftMatrix);
    setBottomRightMatrix(&bottomRightMatrix);
    setTopMatrix(&topMatrix);

    printf("done\n");
    printf("Creating triangle...");
    create_triangle(points, amount, 200, buffer, &bottomLeftMatrix, &bottomRightMatrix, &topMatrix);
    printf("done\n");
    printf("Scaling to image...");
    uint8_t* image_array = scale_to_image(points, amount, width, height);
    printf("done\n");
    
    
    // Save the image
    save_black_white_image(image_array, width, height);
    
    // Free the memory
    free(image_array);
    free(points);
    free(buffer);
    
    return 0;
}