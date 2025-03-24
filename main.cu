#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <vector>

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

void save_image_array(uint8_t* image_array, int width, int height, int channels, int iteration) {
    char filename[50];
    snprintf(filename, 50, "vid_imgs/iteration_%d.ppm", iteration);
    FILE *imageFile = fopen(filename, "wb");
    if (imageFile == NULL) {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    fprintf(imageFile, "P6\n%d %d\n255\n", width, height);
    fwrite(image_array, 1, width * height * channels, imageFile);
    fclose(imageFile);
}

void save_black_white_image(uint8_t* image_array, int width, int height, int iteration) {
    uint8_t* color_image = (uint8_t*)malloc(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        color_image[i * 3] = image_array[i];
        color_image[i * 3 + 1] = image_array[i];
        color_image[i * 3 + 2] = image_array[i];
    }
    save_image_array(color_image, width, height, 3, iteration);
    free(color_image);
}

uint8_t* scale_to_image(point* points, int amount, int width, int height) {
    uint8_t* image_array = (uint8_t*)calloc(width * height, sizeof(uint8_t));
    for (int i = 0; i < amount; i++) {
        int x = fminf(points[i].x * width, width - 1);
        int y = fminf(height - (points[i].y * height), height - 1);
        image_array[y * width + x] = 255;
    }
    return image_array;
}

__global__ void create_triangle_kernel(point* points, int amount, int iterations, int seed, point* all_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= amount) return;

    curandState state;
    curand_init(seed + idx, 0, 0, &state);

    for (int j = 0; j < iterations; j++) {
        point random_trig_point = get_random_trig_point_gpu(&state);
        points[idx].x = 0.5 * (points[idx].x + random_trig_point.x);
        points[idx].y = 0.5 * (points[idx].y + random_trig_point.y);
        all_points[j * amount + idx] = points[idx];
    }
}

void create_triangle_gpu(point* points, int amount, int iterations, int width, int height) {
    point* d_points;
    point* d_all_points;
    cudaMalloc(&d_points, amount * sizeof(point));
    cudaMalloc(&d_all_points, amount * iterations * sizeof(point));
    cudaMemcpy(d_points, points, amount * sizeof(point), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (amount + threadsPerBlock - 1) / threadsPerBlock;
    
    create_triangle_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_points, amount, iterations, time(NULL), d_all_points);
    cudaMemcpy(points, d_points, amount * sizeof(point), cudaMemcpyDeviceToHost);

    point* all_points = (point*)malloc(amount * iterations * sizeof(point));
    cudaMemcpy(all_points, d_all_points, amount * iterations * sizeof(point), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < iterations; i++) {
        uint8_t* image_array = scale_to_image(&all_points[i * amount], amount, width, height);
        save_black_white_image(image_array, width, height, i);
        free(image_array);
    }

    free(all_points);
    cudaFree(d_points);
    cudaFree(d_all_points);
}

// to get video use ffmpeg -framerate 5 -i vid_imgs/iteration_%d.ppm -c:v libx264 -pix_fmt yuv420p output.mp4
int main() {
    int amount = 50000000;
    int iterations = 20;
    int width = 1000;
    int height = 1000;

    point* points = (point*)malloc(amount * sizeof(point));
    for (int i = 0; i < amount; i++) {
        points[i].x = (float)rand() / RAND_MAX;
        points[i].y = (float)rand() / RAND_MAX;
    }

    create_triangle_gpu(points, amount, iterations, width, height);
    printf("GPU computation done! Iteration images saved.\n");

    free(points);
    return 0;
}