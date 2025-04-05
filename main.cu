#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include "matrix.h"
#include <curand.h>
#include <curand_kernel.h>

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

__device__ Matrix<3,3>* get_random_trig_point_gpu(curandState* state, Matrix<3,3>* bottomLeftMatrix, Matrix<3,3>* bottomRightMatrix, Matrix<3,3>* topMatrix) {
    float random = curand_uniform(state);
    if (random < 1.0f / 3.0f) {
        return bottomLeftMatrix;
    } else if (random < 2.0f / 3.0f) {
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

__global__ void create_triangle_gpu_kernel(Matrix<3, 1>* points, int amount, int iterations, Matrix<3, 1>* buffer ,Matrix<3,3> bottomLeftMatrix, Matrix<3,3> bottomRightMatrix, Matrix<3,3> topMatrix, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= amount) {
        return;
    }

    curandState state;
    curand_init(seed + idx, 0, 0, &state);
    Matrix<3, 3>* random_trig_point;
    Matrix<3, 1>* original_point = points + idx;




    Matrix<3, 1>* current_point = original_point;
    Matrix<3, 1>* buffer_point = buffer + idx;

    Matrix<3, 1>* temp = current_point;

    for (int j = 0; j < iterations; j++) {
        random_trig_point = get_random_trig_point_gpu(&state, &bottomLeftMatrix, &bottomRightMatrix, &topMatrix);
        random_trig_point->mult(current_point, buffer_point);
        // we move the pointers around instead of copying the data
        current_point = buffer_point;
        buffer_point = temp;
        temp = current_point;
    }
    if (iterations % 2 != 0) {
        // we have a uneven amount of operations done and such, te last result is in the buffer and not in points.
        // so we copy it over
        *original_point = *buffer_point;
    }
}

void save_black_white_image_with_name(uint8_t* image_array, int width, int height, const char* filename) {
    uint8_t* color_image = (uint8_t*)malloc(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        color_image[i * 3] = image_array[i];
        color_image[i * 3 + 1] = image_array[i];
        color_image[i * 3 + 2] = image_array[i];
    }

    FILE *imageFile = fopen(filename, "wb");
    if (imageFile == NULL) {
        perror("Cannot open output file");
        exit(EXIT_FAILURE);
    }
    fprintf(imageFile, "P6\n%d %d\n255\n", width, height);
    fwrite(color_image, 1, width * height * 3, imageFile);
    fclose(imageFile);
    free(color_image);
}


// RNG Setup Kernel (now fixed)
__global__ void setup_rng_kernel(curandState* states, int amount, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= amount) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// GPU Kernel using persistent RNG states
__global__ void create_triangle_gpu_kernel(
    Matrix<3, 1>* input,
    int amount,
    Matrix<3, 1>* output,
    Matrix<3,3> bottomLeftMatrix,
    Matrix<3,3> bottomRightMatrix,
    Matrix<3,3> topMatrix,
    curandState* rng_states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= amount) return;

    curandState* state = &rng_states[idx];

    Matrix<3,3>* random_trig_point;
    Matrix<3,1>* current_point = input + idx;
    Matrix<3,1>* buffer_point = output + idx;

    random_trig_point = get_random_trig_point_gpu(state, &bottomLeftMatrix, &bottomRightMatrix, &topMatrix);
    random_trig_point->mult(current_point, buffer_point);
}

// Main function with fixed RNG setup
void create_triangle_gpu_with_frames(Matrix<3, 1>* host_points, int amount, int iterations,
    const Matrix<3,3>& bottomLeftMatrix,
    const Matrix<3,3>& bottomRightMatrix,
    const Matrix<3,3>& topMatrix,
    int width, int height)
{
    // Allocate GPU memory
    Matrix<3, 1>* gpu_a;
    Matrix<3, 1>* gpu_b;
    cudaMalloc(&gpu_a, amount * sizeof(Matrix<3, 1>));
    cudaMalloc(&gpu_b, amount * sizeof(Matrix<3, 1>));

    // Copy initial points
    cudaMemcpy(gpu_a, host_points, amount * sizeof(Matrix<3, 1>), cudaMemcpyHostToDevice);

    // Allocate and setup RNG states
    curandState* d_states;
    cudaMalloc(&d_states, amount * sizeof(curandState));

    int threadsPerBlock = 256;
    int blocksPerGrid = (amount + threadsPerBlock - 1) / threadsPerBlock;

    // ✅ Pass `amount` to RNG setup kernel
    setup_rng_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, amount, time(NULL));
    cudaDeviceSynchronize();

    // Buffer for saving frames
    Matrix<3, 1>* save_buffer = (Matrix<3, 1>*)malloc(amount * sizeof(Matrix<3, 1>));
    bool using_a_as_input = true;

    for (int j = 0; j < iterations; j++) {
        Matrix<3, 1>* input = using_a_as_input ? gpu_a : gpu_b;
        Matrix<3, 1>* output = using_a_as_input ? gpu_b : gpu_a;

        create_triangle_gpu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            input, amount, output,
            bottomLeftMatrix, bottomRightMatrix, topMatrix,
            d_states
        );
        cudaDeviceSynchronize();

        // Copy for frame
        cudaMemcpy(save_buffer, output, amount * sizeof(Matrix<3, 1>), cudaMemcpyDeviceToHost);

        // Save image
        uint8_t* image_array = scale_to_image(save_buffer, amount, width, height);
        char filename[64];
        snprintf(filename, sizeof(filename), "./vid_imgs/frame_%03d.ppm", j);
        save_black_white_image_with_name(image_array, width, height, filename);
        free(image_array);

        using_a_as_input = !using_a_as_input;
    }

    // Copy final result
    Matrix<3, 1>* final_output = using_a_as_input ? gpu_b : gpu_a;
    cudaMemcpy(host_points, final_output, amount * sizeof(Matrix<3, 1>), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(d_states);
    free(save_buffer);
}

void create_triangle_gpu(Matrix<3, 1>* points, int amount, int iterations , const Matrix<3,3>& bottomLeftMatrix, const Matrix<3,3>& bottomRightMatrix, const Matrix<3,3>& topMatrix) {
    Matrix<3, 1>* gpu_points;
    Matrix<3, 1>* gpu_buffer;

    cudaMalloc(&gpu_points, amount * sizeof(Matrix<3, 1>));
    cudaMalloc(&gpu_buffer, amount * sizeof(Matrix<3, 1>));

    cudaMemcpy(gpu_points, points, amount * sizeof(Matrix<3, 1>), cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (amount + threadsPerBlock - 1) / threadsPerBlock;

    create_triangle_gpu_kernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_points, amount, iterations, gpu_buffer, bottomLeftMatrix, bottomRightMatrix, topMatrix, time(NULL));
    cudaDeviceSynchronize();
    cudaMemcpy(points, gpu_points, amount * sizeof(Matrix<3, 1>), cudaMemcpyDeviceToHost);
    cudaFree(gpu_points);
    cudaFree(gpu_buffer);

}


int main() {
    srand(1000);
    int width = 500;
    int height = 500;
    int image_size = width * height;
    
    // Generate random points
    int amount = 1000000;
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
    //create_triangle(points, amount, 200, buffer, &bottomLeftMatrix, &bottomRightMatrix, &topMatrix);
    // create_triangle_gpu(points, amount, 20, bottomLeftMatrix, bottomRightMatrix, topMatrix);
    create_triangle_gpu_with_frames(points, amount, 20, bottomLeftMatrix, bottomRightMatrix, topMatrix, width, height);
    printf("done\n");
    printf("Scaling to image...");
    uint8_t* image_array = scale_to_image(points, amount, width, height);
    printf("done\n");
    
    
    // Save the image
    save_black_white_image(image_array, width, height);

    system("ffmpeg -y -framerate 5 -i ./vid_imgs/frame_%03d.ppm -c:v libx264 -pix_fmt yuv420p output.mp4");
    
    // Free the memory
    free(image_array);
    free(points);
    free(buffer);
    
    return 0;
}