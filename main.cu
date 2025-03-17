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


Matrix get_bottom_left_matrix() {
    return Matrix::fromVect({
        {0.5, 0, 0},
        {0, 0.5, 0},
        {0, 0, 1},
    });
}

Matrix get_bottom_right_matrix() {
    return Matrix::fromVect({
        {0.5, 0, 0.5},
        {0, 0.5, 0},
        {0, 0, 1},
    });
}

Matrix get_top_matrix() {
    return Matrix::fromVect({
        {0.5, 0, 0.25},
        {0, 0.5, 0.5},
        {0, 0, 1},
    });
}



Matrix get_random_trig_point() {
    int random = rand() % 9;
    if (random < 5) {
        return get_bottom_left_matrix();
    } else if (random < 7) {
        return get_bottom_right_matrix();
    } else {
        return get_top_matrix();
    }
}



void create_triangle(Matrix* points, int amount, int iterations) {
    // the tirnalgle is on a one by one grid
    for (int j = 0; j < iterations; j++) {
        for (int i = 0; i < amount; i++) {
            Matrix random_trig_point = get_random_trig_point();
            Matrix current_point = points[i];
            Matrix new_point = random_trig_point.mult(current_point);
            points[i] = new_point;
        }
    }

}

Matrix* generate_random_points(int amount) {
    Matrix* points = (Matrix*)malloc(amount * sizeof(Matrix));
    for (int i = 0; i < amount; i++) {
        points[i] = Matrix::fromVect({
            {(float)rand() / RAND_MAX},
            {(float)rand() / RAND_MAX},
            {(float)1.0},
        });
    }
    return points;
}


uint8_t* scale_to_image(Matrix* points, int amount, int width, int height) {
    uint8_t* image_array = (uint8_t*)calloc(width * height, sizeof(uint8_t));

    for (int i = 0; i < amount; i++) {
        Matrix current_point = points[i];
        int x = fminf(current_point.get(0, 0) * width, width - 1);
        int y = fminf(height - current_point.get(1, 0)* height, height - 1);
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
    int amount = 10000;
    printf("Generating random points...\n");
    Matrix* points = generate_random_points(amount);
    printf("done\n");
    printf("Creating triangle...");
    create_triangle(points, amount, 200);
    printf("done\n");
    printf("Scaling to image...");
    uint8_t* image_array = scale_to_image(points, amount, width, height);
    printf("done\n");
    
    
    // Save the image
    save_black_white_image(image_array, width, height);
    
    // Free the memory
    free(image_array);
    free(points);
    
    return 0;
}