#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

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


struct point {
    float x;
    float y;
};

#define BOTTOM_LEFT_POINT {0.0, 0.0}
#define BOTTOM_RIGHT_POINT {1.0, 0.0}
#define TOP_POINT {0.5, 1.0}

__device__ point get_random_trig_point_gpu(curandState* state) {
    float random = curand_uniform(state);
    if (random < 1.0f / 3.0f) {
        return BOTTOM_LEFT_POINT;
    } else if (random < 2.0f / 3.0f) {
        return BOTTOM_RIGHT_POINT;
    } else {
        return TOP_POINT;
    }
}

point get_random_trig_point() {
    int random = rand() % 3;
    if (random == 0) {
        return BOTTOM_LEFT_POINT;
    } else if (random == 1) {
        return BOTTOM_RIGHT_POINT;
    } else {
        return TOP_POINT;
    }
}


void create_triangle(point* points, int amount, int iterations) {
    // the triangle is on a one by one grid

    for (int j = 0; j < iterations; j++) {
        for (int i = 0; i < amount; i++) {
            point random_trig_point = get_random_trig_point();
            point* current_point = &points[i];
            current_point->x = 0.5 * (current_point->x + random_trig_point.x);
            current_point->y = 0.5 * (current_point->y + random_trig_point.y);
        }
    }

}

point* generate_random_points(int amount) {
    point* points = (point*)malloc(amount * sizeof(point));
    for (int i = 0; i < amount; i++) {
        points[i].x = (float)rand() / RAND_MAX;
        points[i].y = (float)rand() / RAND_MAX;
    }
    return points;
}


uint8_t* scale_to_image(point* points, int amount, int width, int height) {
    uint8_t* image_array = (uint8_t*)malloc(width * height);
    // first set all to zero
    for (int i = 0; i < width * height; i++) {
        image_array[i] = 0;
    }

    for (int i = 0; i < amount; i++) {
        point current_point = points[i];
        // int x = current_point.x * width;
        // int y = height - current_point.y * height;
        int x = fminf(current_point.x * width, width - 1);
        int y = fminf(height - (current_point.y * height), height - 1);
        image_array[y * width + x] = 255;
    }
    return image_array;
}

// GPU version
__global__ void create_triangle_kernel(point* points, int amount, int iterations, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= amount) return;

    curandState state;
    curand_init(seed + idx, 0, 0, &state);

    for (int j = 0; j < iterations; j++) {
        // int rand_val = curand(&state) % 3;
        point random_trig_point = get_random_trig_point_gpu(&state);
        points[idx].x = 0.5 * (points[idx].x + random_trig_point.x);
        points[idx].y = 0.5 * (points[idx].y + random_trig_point.y);
    }
}

void create_triangle_gpu(point* points, int amount, int iterations) {
    point* d_points;
    cudaMalloc(&d_points, amount * sizeof(point));
    cudaMemcpy(d_points, points, amount * sizeof(point), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (amount + threadsPerBlock - 1) / threadsPerBlock;
    
    create_triangle_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_points, amount, iterations, time(NULL));
    cudaMemcpy(points, d_points, amount * sizeof(point), cudaMemcpyDeviceToHost);
    
    cudaFree(d_points);
}


int main() {
    // // Define the image dimensions
    // srand(1000);
    // int width = 300;
    // int height = 300;
    // int image_size = width * height;
    
    // // Generate random points
    // int amount = 100000;
    // point* points = generate_random_points(amount);
    // create_triangle(points, amount, 1000);
    // uint8_t* image_array = scale_to_image(points, amount, width, height);

    // printf("done\n");
    
    
    // // Save the image
    // save_black_white_image(image_array, width, height);
    
    // // Free the memory
    // free(image_array);
    // free(points);
    
    // return 0;

    int amount = 100000000;
    int iterations = 10000;
    int width = 1000;
    int height = 1000;

    point* points = (point*)malloc(amount * sizeof(point));
    for (int i = 0; i < amount; i++) {
        points[i].x = (float)rand() / RAND_MAX;
        points[i].y = (float)rand() / RAND_MAX;
    }

    create_triangle_gpu(points, amount, iterations);

    printf("GPU computation done!\n");

    // scale and save the image
    uint8_t* image_array = scale_to_image(points, amount, width, height);
    save_black_white_image(image_array, width, height);

    // Free the memory
    free(points);
    free(image_array);
    return 0;
}