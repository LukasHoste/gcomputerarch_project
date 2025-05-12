#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include "matrix.h"
#include <curand.h>
#include <curand_kernel.h>
#include "stable_random.h"
//#include <algorithm>
#include <array>
#include <random>
#include <chrono>

# define OUR_PI		3.14159265358979323846	/* pi */
# define OUR_E        2.71828182845904523536	/* e */
StableRandom stable_random;
#define WIDTH 1920
#define HEIGHT 1920
#define CHANNELS 3

struct ColoredPoint {
    Matrix<3, 1> pos;
    Matrix<4, 1> color;
};

template <typename T>
__host__ __device__ inline T my_clamp(const T& val, const T& lo, const T& hi) {
    return (val < lo) ? lo : (val > hi) ? hi : val;
}

__constant__ char global_matrixes_data[sizeof(Matrix<3, 3>) * 3 + sizeof(Matrix<4, 4>) * 3];

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


Matrix<3, 3> create_random_scale_matrix() {
    Matrix<3, 3> matrix;
    double data[3][3] = {
        {(double)stablerand_next(&stable_random) *2 - 1, 0, 0},
        {0, (double)stablerand_next(&stable_random) * 2 - 1, 0},
        {0, 0, 1},
    };
    matrix.setData(data);
    return matrix;
}
Matrix<3, 3> create_random_translation_matrix() {
    Matrix<3, 3> matrix;
    double data[3][3] = {
        {1, 0, stablerand_next(&stable_random) * 2 -1},
        {0, 1, stablerand_next(&stable_random) * 2 - 1},
        {0, 0, 1},
    };
    matrix.setData(data);
    return matrix;
}

Matrix<3, 3> create_random_rotation_matrix() {
    Matrix<3, 3> matrix;
    double angle = ((double)stablerand_next(&stable_random)) * 2 * OUR_PI - OUR_PI;
    double data[3][3] = {
        {cos(angle), -sin(angle), 0},
        {sin(angle), cos(angle), 0},
        {0, 0, 1},
    };
    matrix.setData(data);
    return matrix;
}
Matrix<3, 3> create_random_shear_matrix() {
    Matrix<3, 3> matrix;
    double data[3][3] = {
        {1, (double)stablerand_next(&stable_random) * 2 - 1, 0},
        {(double)stablerand_next(&stable_random) * 2 -1, 1, 0},
        {0, 0, 1},
    };
    matrix.setData(data);
    return matrix;
}
Matrix<3, 3> create_random_affine_matrix() {
    return create_random_scale_matrix() * create_random_rotation_matrix()* create_random_translation_matrix();
}

// https://lisyarus.github.io/blog/posts/transforming-colors-with-matrices.html
Matrix<3, 3> create_random_darkening_matrix() {
    Matrix<3, 3> matrix;
    double data[3][3] = {
        {1 - (double)stablerand_next(&stable_random), 0, 0},
        {0, 1 - (double)stablerand_next(&stable_random), 0},
        {0, 0, 1 - (double)stablerand_next(&stable_random)}
    };
    matrix.setData(data);
    return matrix;
}

// Generates a small random double between -maxChange and +maxChange
double random_small_change(double maxChange = 0.1) {
    return ((double)stablerand_next(&stable_random)) * 2 * maxChange - maxChange;
}

// random decrease or increase of R, G, B with smaller range
Matrix<3, 3> create_random_decrease_increase_matrix() {
    Matrix<3, 3> matrix;
    double data[3][3] = {
        {1 + random_small_change(0.5), 0, 0},
        {0, 1 + random_small_change(0.5), 0},
        {0, 0, 1 + random_small_change(0.5)}
    };
    matrix.setData(data);
    return matrix;
}

// random color shift matrix e.g. R->G, G->B, B->R or R->B, G->R, B->G
Matrix<3, 3> create_color_shift_matrix_1() {
    Matrix<3, 3> matrix;
    double data[3][3] = {
        {0, 0.8, 0},
        {0, 0, 0.8},
        {0.8, 0, 0}
    };
    matrix.setData(data);
    return matrix;
}

Matrix<3, 3> create_color_shift_matrix_2() {
    Matrix<3, 3> matrix;
    double data[3][3] = {
        {0, 0, 0.8},
        {0, 0.8, 0},
        {0.8, 0, 0}
    };
    matrix.setData(data);
    return matrix;
}

void shuffle_array_in_place(int* arr, int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = (int)(stablerand_next(&stable_random) * (i + 1));
        if (j != i) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
}


// assume Matrix<3,3> has a setData(double[3][3]) member
Matrix<3, 3> create_random_color_shift(bool excludeIdentity = true) {
    // permutation array: index = destination channel 0=R,1=G,2=B
    // value = source channel to map from

    int perm[3] = {0, 1, 2};
    shuffle_array_in_place(perm, 3);
   
    
    // build the scaled permutation matrix
    double data[3][3] = {{0}};
    for (int dst = 0; dst < 3; ++dst) {
        int src = perm[dst];
        // Use your custom random function to scale between 0 and 2
        double scale = stablerand_next(&stable_random) * 2.0; // Replace with your function
        data[dst][src] = scale;
    }
    
    Matrix<3,3> M;
    M.setData(data);
    return M;
}

// lighten color
Matrix<3, 3> create_random_lighten_matrix() {
    Matrix<3, 3> matrix;
    double data[3][3] = {
        {1 + (double)stablerand_next(&stable_random), 0, 0},
        {0, 1 + (double)stablerand_next(&stable_random), 0},
        {0, 0, 1 + (double)stablerand_next(&stable_random)}
    };
    matrix.setData(data);
    return matrix;
}



Matrix<3,3>* get_random_point(Matrix<3,3>* bottomLeftMatrix, Matrix<3,3>* bottomRightMatrix, Matrix<3,3>* topMatrix) {
    int random = (int)floor(stablerand_next(&stable_random) * 81) % 3;
    if (random < 1) {
        return bottomLeftMatrix;
    } else if (random < 2) {
        return bottomRightMatrix;
    } else {
        return topMatrix;
    }
}

__device__ Matrix<3,3>* get_random_point_gpu(curandState* state, Matrix<3,3>* bottomLeftMatrix, Matrix<3,3>* bottomRightMatrix, Matrix<3,3>* topMatrix) {
    float random = curand_uniform(state);
    if (random < 1.0f / 3.0f) {
        return bottomLeftMatrix;
    } else if (random < 2.0f / 3.0f) {
        return bottomRightMatrix;
    } else {
        return topMatrix;
    }
}

__device__ Matrix<4,4>* get_random_point_gpu(curandState* state, Matrix<4,4>* bottomLeftMatrix, Matrix<4,4>* bottomRightMatrix, Matrix<4,4>* topMatrix) {
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
            Matrix<3, 3>* random_trig_point = get_random_point(bottomLeftMatrix, bottomRightMatrix, topMatrix);
            Matrix<3,1>* current_point = points + i;
            Matrix<3,1>* buffer_point = buffer + i;
            random_trig_point->mult(current_point, buffer_point);
            *current_point = *buffer_point;
        }
    }

}


// between 0 and one in all directions
void rescale_points(ColoredPoint* points, int amount) {
    double min_x = 1;
    double min_y = 1;
    double max_x = 0;
    double max_y = 0;
    for (int i = 0; i < amount; i++) {
        ColoredPoint current_point = points[i];
        if (current_point.pos.at(0, 0) < min_x) {
            min_x = current_point.pos.at(0, 0);
        }
        if (current_point.pos.at(1, 0) < min_y) {
            min_y = current_point.pos.at(1, 0);
        }
        if (current_point.pos.at(0, 0) > max_x) {
            max_x = current_point.pos.at(0, 0);
        }
        if (current_point.pos.at(1, 0) > max_y) {
            max_y = current_point.pos.at(1, 0);
        }
    }
    for (int i = 0; i < amount; i++) {
        ColoredPoint current_point = points[i];
        current_point.pos.at(0, 0) = (current_point.pos.at(0, 0) - min_x) / (max_x - min_x);
        current_point.pos.at(1, 0) = (current_point.pos.at(1, 0) - min_y) / (max_y - min_y);
        points[i] = current_point;
    }
}

uint8_t* scale_to_image(ColoredPoint* points, int amount, int width, int height) {
    int pointsPerPixel = amount / (width * height);
    printf("pointsPerPixel: %d\n", pointsPerPixel);
    if (pointsPerPixel < 1) {
        pointsPerPixel = 1;
    }



    uint8_t* image_array = (uint8_t*)calloc(width * height * CHANNELS, sizeof(uint8_t));

    for (int i = 0; i < amount; i++) {
        ColoredPoint current_point = points[i];
        int x = fminf(current_point.pos.at(0, 0) * width, width - 1);
        int y = fminf(height - current_point.pos.at(1, 0)* height, height - 1);
        if (x < 0) {
            x = 0;
        }
        if (y < 0) {
            y = 0;
        }
        uint rIndex = (y * width + x) * 3;
        uint gIndex = rIndex + 1;
        uint bIndex = gIndex + 1;

        uint rValue = (uint)(current_point.color.at(0, 0) * 255);
        uint gValue = (uint)(current_point.color.at(1, 0) * 255);
        uint bValue = (uint)(current_point.color.at(2, 0) * 255);
    
        image_array[rIndex] = min(255, image_array[rIndex] + rValue / pointsPerPixel);
        image_array[gIndex] = min(255, image_array[gIndex] + gValue / pointsPerPixel);
        image_array[bIndex] = min(255, image_array[bIndex] + bValue / pointsPerPixel);
    }
    return image_array;
}

__global__ void create_triangle_gpu_kernel(ColoredPoint* points, int amount, int iterations, ColoredPoint* buffer, curandState* rng_states) {
    Matrix<3, 3>* global_matrixes = (Matrix<3, 3>*)global_matrixes_data;
    char* raw_color_matrixes = global_matrixes_data + 3 * sizeof(Matrix<3, 3>);

    Matrix<4, 4>* global_color_matr = (Matrix<4, 4>*) raw_color_matrixes;

    // striding
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < amount; idx += blockDim.x * gridDim.x) {
        ColoredPoint* og_buffer = buffer + idx;


        curandState* state = &rng_states[idx];
        Matrix<3, 3>* random_pos_matrix;
        Matrix<4, 4>* random_color_matrix;
        ColoredPoint* original_point = points + idx;

        ColoredPoint* current_point = original_point;
        ColoredPoint* buffer_point = buffer + idx;

        ColoredPoint* temp = current_point;

        for (int j = 0; j < iterations; j++) {
            random_pos_matrix = get_random_point_gpu(state, &global_matrixes[0], &global_matrixes[1], &global_matrixes[2]);
            random_color_matrix = get_random_point_gpu(state, &global_color_matr[0], &global_color_matr[1], &global_color_matr[2]);
            //Matrix<3, 1> current_pos = current_point->pos;
            //buffer_point->pos = current_pos;

            random_pos_matrix->mult(&current_point->pos, &buffer_point->pos);
            random_color_matrix->mult(&current_point->color, &buffer_point->color);

            // color correcting
            /*buffer_point->color.at(2, 0) = 1.0;
            for(int i = 0; i < 2; i++) {
                buffer_point->color.at(i, 0) = fminf(fmaxf(buffer_point->color.at(i, 0), 0.0f), 1.0f);
            }*/

            // we move the pointers around instead of copying the data
            temp = current_point;
            current_point = buffer_point;
            buffer_point = temp;
        }
        /*og_buffer = *current_point;
        if (iterations % 2 != 0) {
            // we have a even amount of operations done and such, te last result is in the points and not in the buffer.
            // so we copy it over
            *og_buffer = *current_point;
        }*/
    }
}

void save_image_array(const char* filename, uint8_t* image_array) {
    FILE *imageFile = fopen(filename, "wb");
    if (!imageFile) {
        perror("Cannot open output file");
        exit(EXIT_FAILURE);
    }
    fprintf(imageFile, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(image_array, 1, WIDTH * HEIGHT * CHANNELS, imageFile);
    fclose(imageFile);
}

ColoredPoint* generate_random_points(int amount) {
    ColoredPoint* points = new ColoredPoint[amount];
    for (int i = 0; i < amount; i++) {
        double pos_data[3][1] = {
            {(double)stablerand_next(&stable_random)},
            {(double)stablerand_next(&stable_random)},
            {1.0}
        };
        points[i].pos.setData(pos_data);
        
        // Start with white color
        double color_data[4][1] = {{(double)stablerand_next(&stable_random)}, {(double)stablerand_next(&stable_random)}, {(double)stablerand_next(&stable_random)}, {(double)stablerand_next(&stable_random)}};
        points[i].color.setData(color_data);
    }
    return points;
}

void save_image_with_name(uint8_t* image_array, int width, int height, const char* filename) {
    uint8_t* color_image = (uint8_t*)malloc(width * height * 3);
    for (int i = 0; i < width * height * CHANNELS; i++) {
        color_image[i] = image_array[i];
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

__device__ float fatomicMin(float *addr, float value)

{

        float old = *addr, assumed;

        if(old <= value) return old;

        do

        {

                assumed = old;

                old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));

        }while(old!=assumed);

        return old;

}

__device__ float fatomicMax(float *addr, float value)

{

        float old = *addr, assumed;

        if(old >= value) return old;

        do

        {

                assumed = old;

                old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));

        }while(old!=assumed);

        return old;

}


// RNG Setup Kernel (now fixed)
__global__ void setup_rng_kernel(curandState* states, int amount, int seed) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < amount; idx += blockDim.x * gridDim.x) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void create_image_floaty_gpu_kernel(ColoredPoint* points, int amount, float* image_data, int width, int height,
    float* min_x, float* min_y, float* max_x, float* max_y,
    float* min_r, float* min_g, float* min_b, float* max_r, float* max_g, float* max_b
) {
    // striding
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < amount; idx += blockDim.x * gridDim.x) {
        ColoredPoint* current_point = points + idx;
        float x_f_norm = ((float) current_point->pos.at(0, 0) - *min_x) / (*max_x - *min_x);
        float y_f_nrom = ((float) current_point->pos.at(1, 0) - *min_y) / (*max_y - *min_y);

        float r_f_norm = ((float) current_point->color.at(0, 0) - *min_r) / (*max_r - *min_r);
        float g_f_norm = ((float) current_point->color.at(1, 0) - *min_g) / (*max_g - *min_g);
        float b_f_norm = ((float) current_point->color.at(2, 0) - *min_b) / (*max_b - *min_b);


        float x_f = fminf(x_f_norm * width, width - 1);
        float y_f = fminf(height - y_f_nrom * height, height - 1);
        if (x_f < 0) {
            x_f = 0;
        }
        if (y_f < 0) {
            y_f = 0;
        }

        int x_i = (int)x_f;
        int y_i = (int)y_f;

        float dx = x_f - x_i;
        float dy = y_f - y_i;

        float w1 = (1.0f - dx) * (1.0f - dy);
        float w2 = dx * (1.0f - dy);
        float w3 = (1.0f - dx) * dy;
        float w4 = dx * dy;

        int x0 = x_i;
        int y0 = y_i;
        int x1 = min(x_i + 1, width - 1);
        int y1 = min(y_i + 1, height - 1);
        float mult = 1;

        uint firstIndexR = (y0 * width + x0) * 3;
        uint firstIndexG = firstIndexR + 1;
        uint firstIndexB = firstIndexG + 1;

        uint secondIndexR = (y0 * width + x1) * 3;
        uint secondIndexG = secondIndexR + 1;
        uint secondIndexB = secondIndexG + 1;

        uint thirdIndexR = (y1 * width + x0) * 3;
        uint thirdIndexG = thirdIndexR + 1;
        uint thirdIndexB = thirdIndexG + 1;
        
        uint fourthIndexR = (y1 * width + x1) * 3;
        uint fourthIndexG = fourthIndexR + 1;
        uint fourthIndexB = fourthIndexG + 1;

        float rValue = r_f_norm;
        float gValue = g_f_norm;
        float bValue = b_f_norm;

        //rValue = 1;
        //gValue = 1;
        //bValue = 1;
        //w1 = 1;
        //w2 = 1;
        //w3 = 1;
        //w4 = 1;


        // atomic adds

        atomicAdd(&image_data[firstIndexR], w1 * rValue * mult);
        atomicAdd(&image_data[firstIndexG], w1 * gValue * mult);
        atomicAdd(&image_data[firstIndexB], w1 * bValue * mult);

        atomicAdd(&image_data[secondIndexR], w2 * rValue * mult);
        atomicAdd(&image_data[secondIndexG], w2 * gValue * mult);
        atomicAdd(&image_data[secondIndexB], w2 * bValue * mult);

        atomicAdd(&image_data[thirdIndexR], w3 * rValue * mult);
        atomicAdd(&image_data[thirdIndexG], w3 * gValue * mult);
        atomicAdd(&image_data[thirdIndexB], w3 * bValue * mult);

        atomicAdd(&image_data[fourthIndexR], w4 * rValue * mult);
        atomicAdd(&image_data[fourthIndexG], w4 * gValue * mult);
        atomicAdd(&image_data[fourthIndexB], w4 * bValue * mult);
    }
}
__device__ float mapFunction(float value, float pin) {
    //return value * 2 - value * value;
    //float o = powf(value, 0.1) * 0.4;
    float o = (1 - exp((-value *0.3 ))) * 0.9;
    //float o = (1 - exp(((value * 2) / pin) * logf(1- OUR_E / 3)))*1.01;
    //return value;
    if (o * 255 > 255) {
        return 1;
    }
    return o;
}


__global__ void float_to_uint8_kernel(float* input, uint8_t* output, unsigned long amount, float* min_r, float* min_g, float* min_b, float* max_r, float* max_g, float* max_b) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < amount; idx += blockDim.x * gridDim.x) {
        float r = input[idx * 3];
        float g = input[idx * 3 + 1];
        float b = input[idx * 3 + 2];

        // clamped
        //r = (r - *min_r) / (*max_r - *min_r);
        //g = (g - *min_g) / (*max_g - *min_g);
        //b = (b - *min_b) / (*max_b - *min_b);



        output[idx * 3] = (uint8_t)(mapFunction(r, *max_r) * 255);
        output[idx * 3 + 1] = (uint8_t)(mapFunction(g, *max_g) * 255);
        output[idx * 3 + 2] = (uint8_t)(mapFunction(b, *max_b) * 255);
    }
}




__global__ void get_scaling_params_kernel(ColoredPoint* points, int amount, float* min_x, float* min_y, float* max_x, float* max_y,
    float* min_r, float* min_g, float* min_b, float* max_r, float* max_g, float* max_b) {
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < amount; idx += blockDim.x * gridDim.x) {
        ColoredPoint* current_point = points + idx;
        fatomicMin(min_x, (float) current_point->pos.at(0, 0));
        fatomicMin(min_y, (float) current_point->pos.at(1, 0));
        fatomicMax(max_x, (float) current_point->pos.at(0, 0));
        fatomicMax(max_y, (float) current_point->pos.at(1, 0));
        
        fatomicMin(min_r, (float) current_point->color.at(0, 0));
        fatomicMin(min_g, (float) current_point->color.at(1, 0));
        fatomicMin(min_b, (float) current_point->color.at(2, 0));
        fatomicMax(max_r, (float) current_point->color.at(0, 0));
        fatomicMax(max_g, (float) current_point->color.at(1, 0));
        fatomicMax(max_b, (float) current_point->color.at(2, 0));
    }
}

__global__ void get_color_scaling_params(float* rgbValues, int amount, float* min_r, float* min_g, float* min_b, float* max_r, float* max_g, float* max_b) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < amount; idx += blockDim.x * gridDim.x) {
        fatomicMin(min_r, rgbValues[idx * 3]);
        fatomicMin(min_g, rgbValues[idx * 3 + 1]);
        fatomicMin(min_b, rgbValues[idx * 3 + 2]);
        fatomicMax(max_r, rgbValues[idx * 3]);
        fatomicMax(max_g, rgbValues[idx * 3 + 1]);
        fatomicMax(max_b, rgbValues[idx * 3 + 2]);
    }
}

// Main function with fixed RNG setup
void create_triangle_gpu_with_frames(ColoredPoint* host_points, int amount, int iterations,
    int width, int height)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    // Allocate GPU memory
    ColoredPoint* gpu_a;
    ColoredPoint* gpu_b;

    cudaMallocAsync(&gpu_a, amount * sizeof(ColoredPoint), stream);
    cudaMallocAsync(&gpu_b, amount * sizeof(ColoredPoint), stream);

    // Copy initial points
    cudaMemcpyAsync(gpu_a, host_points, amount * sizeof(ColoredPoint), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu_b, host_points, amount * sizeof(ColoredPoint), cudaMemcpyHostToDevice, stream);

    // Allocate and setup RNG states
    curandState* d_states;
    cudaMallocAsync(&d_states, amount * sizeof(curandState), stream);

    int threadsPerBlock = 256;
    int blocksPerGrid = (amount + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = min(blocksPerGrid, 65535); // Limit to 65535 blocks

    // ✅ Pass `amount` to RNG setup kernel
    setup_rng_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_states, amount, time(NULL));
    //cudaDeviceSynchronize();

    float* gpu_floaty_save_buffer = nullptr;
    cudaMallocAsync(&gpu_floaty_save_buffer, width * height * sizeof(float) * 3, stream);
    cudaMemsetAsync(gpu_floaty_save_buffer, 0, width * height * sizeof(float) * 3, stream);

    uint8_t* gpu_save_buffer = nullptr;
    cudaMallocAsync(&gpu_save_buffer, width * height * sizeof(uint8_t) * 3, stream);
    cudaMemsetAsync(gpu_save_buffer, 0, width * height * sizeof(uint8_t) * 3, stream);


    float* scaling_data = nullptr;
    cudaMallocAsync(&scaling_data, 10 * sizeof(float), stream);
    float* min_x = scaling_data;
    float* min_y = scaling_data + 1;
    float* max_x = scaling_data + 2;
    float* max_y = scaling_data + 3;
    
    float* min_r = scaling_data + 4;
    float* min_g = scaling_data + 5;
    float* min_b = scaling_data + 6;

    float* max_r = scaling_data + 7;
    float* max_g = scaling_data + 8;
    float* max_b = scaling_data + 9;

    uint8_t* save_buffer = (uint8_t*)malloc(width * height * sizeof(uint8_t) * 3);

    float fixed_min_start = 1000;
    float fixed_max_start = -1000;

    bool using_a_as_input = true;
    printf("starting \n");
    // changed ordering so that the gpu is busy when the cpu is busy
    for (int j = 0; j < iterations; j++) {
        ColoredPoint* input = using_a_as_input ? gpu_a : gpu_b;
        ColoredPoint* output = using_a_as_input ? gpu_b : gpu_a;



        create_triangle_gpu_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            input, amount, 1, output,
            d_states
        );

        ColoredPoint temp[10];
        cudaMemcpyAsync(temp, output, 10 * sizeof(ColoredPoint), cudaMemcpyDeviceToHost, stream);
        for (int i = 0; i < 10; i++) {
            //printf("temp[%d]: %f %f %f\n", i, temp[i].color.at(0, 0), temp[i].color.at(1, 0), temp[i].color.at(2, 0));
        }

        //cudaDeviceSynchronize();

        cudaMemcpyAsync(min_x, &fixed_min_start, sizeof(float),cudaMemcpyHostToDevice,stream);
        cudaMemcpyAsync(min_y, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(max_x, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(max_y, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice,stream);
        cudaMemcpyAsync(min_r, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(min_g, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(min_b, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(max_r, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(max_g, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(max_b, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        get_scaling_params_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            output, amount, min_x, min_y, max_x, max_y, min_r, min_g, min_b, max_r, max_g, max_b
        );
        float cpu_minr = 1;
        cudaMemcpyAsync(&cpu_minr, min_r, sizeof(float), cudaMemcpyDeviceToHost, stream);
        float cpu_maxr = 0;
        cudaMemcpyAsync(&cpu_maxr, max_r, sizeof(float), cudaMemcpyDeviceToHost, stream);
        float cpu_ming = 1;
        cudaMemcpyAsync(&cpu_ming, min_g, sizeof(float), cudaMemcpyDeviceToHost, stream);
        float cpu_maxg = 0;
        cudaMemcpyAsync(&cpu_maxg, max_g, sizeof(float), cudaMemcpyDeviceToHost, stream);
        float cpu_minb = 1;
        cudaMemcpyAsync(&cpu_minb, min_b, sizeof(float), cudaMemcpyDeviceToHost, stream);
        float cpu_maxb = 0;
        cudaMemcpyAsync(&cpu_maxb, max_b, sizeof(float), cudaMemcpyDeviceToHost, stream);

        printf("cpu_minr: %f\n", cpu_minr);
        printf("cpu_maxr: %f\n", cpu_maxr);
        printf("cpu_ming: %f\n", cpu_ming);
        printf("cpu_maxg: %f\n", cpu_maxg);
        printf("cpu_minb: %f\n", cpu_minb);
        printf("cpu_maxb: %f\n", cpu_maxb);

       

        create_image_floaty_gpu_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            output, amount, gpu_floaty_save_buffer, width, height,
            min_x, min_y, max_x, max_y,
            min_r, min_g, min_b, max_r, max_g, max_b
        );

        cudaMemcpyAsync(min_r, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(min_g, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(min_b, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(max_r, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(max_g, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(max_b, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
        // get the scaling now that all is accumulated!
        get_color_scaling_params<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            gpu_floaty_save_buffer, width * height, min_r, min_g, min_b, max_r, max_g, max_b
        );
        

        float_to_uint8_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            gpu_floaty_save_buffer, gpu_save_buffer, width * height, min_r, min_g, min_b, max_r, max_g, max_b
        );
        cudaMemsetAsync(gpu_floaty_save_buffer, 0, width * height * sizeof(float) * 3, stream);
        printf("starting memcopy\n");
        // Copy for frame
        cudaMemcpyAsync(save_buffer, gpu_save_buffer, width * height * sizeof(uint8_t) * 3, cudaMemcpyDeviceToHost, stream);

        // Save image
        //rescale_points(save_buffer, amount);

        //uint8_t* image_array = scale_to_image(save_buffer, amount, width, height);
        char filename[64];
        snprintf(filename, sizeof(filename), "./vid_imgs/frame_%03d.ppm", j);
        printf("saving image %s\n", filename);
        save_image_with_name(save_buffer, width, height, filename);
        printf("done\n");
        //free(image_array);

        using_a_as_input = !using_a_as_input;
    }

    // Copy final result
    ColoredPoint* final_output = using_a_as_input ? gpu_b : gpu_a;
    cudaMemcpyAsync(host_points, final_output, amount * sizeof(Matrix<3, 1>), cudaMemcpyDeviceToHost, stream);

    // Cleanup
    cudaFreeAsync(gpu_a, stream);
    cudaFreeAsync(gpu_b, stream);
    cudaFreeAsync(d_states, stream);
    cudaFreeAsync(gpu_floaty_save_buffer, stream);
    cudaFreeAsync(gpu_save_buffer, stream);
    cudaFreeAsync(scaling_data, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    free(save_buffer);
}


// Main function with fixed RNG setup
void create_triangle_gpu_single(ColoredPoint* host_points, int amount, int iterations,
    int width, int height, char* output_buffer, float* cpu_measured_time)
{
    auto start = std::chrono::high_resolution_clock::now();
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    // Allocate GPU memory
    ColoredPoint* gpu_a;
    ColoredPoint* gpu_b;

    cudaMallocAsync(&gpu_a, amount * sizeof(ColoredPoint), stream);
    cudaMallocAsync(&gpu_b, amount * sizeof(ColoredPoint), stream);

    // Copy initial points
    cudaMemcpyAsync(gpu_a, host_points, amount * sizeof(ColoredPoint), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu_b, host_points, amount * sizeof(ColoredPoint), cudaMemcpyHostToDevice, stream);

    // Allocate and setup RNG states
    curandState* d_states;
    cudaMallocAsync(&d_states, amount * sizeof(curandState), stream);

    int threadsPerBlock = 256;
    int blocksPerGrid = (amount + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = min(blocksPerGrid, 65535); // Limit to 65535 blocks

    // ✅ Pass `amount` to RNG setup kernel
    setup_rng_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_states, amount, time(NULL));
    //cudaDeviceSynchronize();

    float* gpu_floaty_save_buffer = nullptr;
    cudaMallocAsync(&gpu_floaty_save_buffer, width * height * sizeof(float) * 3, stream);
    cudaMemsetAsync(gpu_floaty_save_buffer, 0, width * height * sizeof(float) * 3, stream);

    uint8_t* gpu_save_buffer = nullptr;
    cudaMallocAsync(&gpu_save_buffer, width * height * sizeof(uint8_t) * 3, stream);
    cudaMemsetAsync(gpu_save_buffer, 0, width * height * sizeof(uint8_t) * 3, stream);


    float* scaling_data = nullptr;
    cudaMallocAsync(&scaling_data, 10 * sizeof(float), stream);
    float* min_x = scaling_data;
    float* min_y = scaling_data + 1;
    float* max_x = scaling_data + 2;
    float* max_y = scaling_data + 3;
    
    float* min_r = scaling_data + 4;
    float* min_g = scaling_data + 5;
    float* min_b = scaling_data + 6;

    float* max_r = scaling_data + 7;
    float* max_g = scaling_data + 8;
    float* max_b = scaling_data + 9;

    float fixed_min_start = 1000;
    float fixed_max_start = -1000;

    // changed ordering so that the gpu is busy when the cpu is busy

    ColoredPoint* input = gpu_a;
    ColoredPoint* output = gpu_b;


    create_triangle_gpu_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        input, amount, iterations, output,
        d_states
    );

    ColoredPoint temp[10];
    cudaMemcpyAsync(temp, output, 10 * sizeof(ColoredPoint), cudaMemcpyDeviceToHost, stream);

    //cudaDeviceSynchronize();

    cudaMemcpyAsync(min_x, &fixed_min_start, sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(min_y, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(max_x, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(max_y, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(min_r, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(min_g, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(min_b, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(max_r, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(max_g, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(max_b, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    get_scaling_params_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        output, amount, min_x, min_y, max_x, max_y, min_r, min_g, min_b, max_r, max_g, max_b
    );

    create_image_floaty_gpu_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        output, amount, gpu_floaty_save_buffer, width, height,
        min_x, min_y, max_x, max_y,
        min_r, min_g, min_b, max_r, max_g, max_b
    );

    cudaMemcpyAsync(min_r, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(min_g, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(min_b, &fixed_min_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(max_r, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(max_g, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(max_b, &fixed_max_start, sizeof(float), cudaMemcpyHostToDevice, stream);
    // get the scaling now that all is accumulated!
    get_color_scaling_params<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        gpu_floaty_save_buffer, width * height, min_r, min_g, min_b, max_r, max_g, max_b
    );
    

    float_to_uint8_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        gpu_floaty_save_buffer, gpu_save_buffer, width * height, min_r, min_g, min_b, max_r, max_g, max_b
    );
    cudaMemsetAsync(gpu_floaty_save_buffer, 0, width * height * sizeof(float) * 3, stream);
    // Copy for frame
    cudaMemcpyAsync(output_buffer, gpu_save_buffer, width * height * sizeof(uint8_t) * 3, cudaMemcpyDeviceToHost, stream);

    // Save image
    //rescale_points(save_buffer, amount);
    save_image_array("./vid_imgs/frame.ppm", (uint8_t*)output_buffer);
    //int8_t* image_array = scale_to_image(save_buffer, amount, width, height);
    //free(image_array);


    // Copy final result
    ColoredPoint* final_output = gpu_b;
    cudaMemcpyAsync(host_points, final_output, amount * sizeof(Matrix<3, 1>), cudaMemcpyDeviceToHost, stream);

    // Cleanup
    cudaFreeAsync(gpu_a, stream);
    cudaFreeAsync(gpu_b, stream);
    cudaFreeAsync(d_states, stream);
    cudaFreeAsync(gpu_floaty_save_buffer, stream);
    cudaFreeAsync(gpu_save_buffer, stream);
    cudaFreeAsync(scaling_data, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;
    *cpu_measured_time = duration_ms.count();
}


void create_triangle_gpu(ColoredPoint* points, int amount, int iterations) {
    ColoredPoint* gpu_points;
    ColoredPoint* gpu_buffer;
    curandState* gpu_rng_states;

    cudaMalloc(&gpu_rng_states, amount * sizeof(curandState));

    cudaMalloc(&gpu_points, amount * sizeof(ColoredPoint));
    cudaMalloc(&gpu_buffer, amount * sizeof(ColoredPoint));

    cudaMemcpy(gpu_points, points, amount * sizeof(ColoredPoint), cudaMemcpyHostToDevice);
    
    
    unsigned long long kb = amount * sizeof(ColoredPoint) / 1024;
    unsigned long long mb = kb / 1024;
    printf("Size of copy: %llu KB\n", kb);
    printf("Size of copy: %llu MB\n", mb);
    int threadsPerBlock = 256;
    int blocksPerGrid = (amount + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = min(blocksPerGrid, 65535); // Limit to 65535 blocks
    printf("blocksPerGrid: %d\n", blocksPerGrid);
    setup_rng_kernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_rng_states, amount, time(NULL));

    create_triangle_gpu_kernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_points, amount, iterations, gpu_buffer, gpu_rng_states);
    cudaDeviceSynchronize();
    cudaMemcpy(points, gpu_buffer, amount * sizeof(ColoredPoint), cudaMemcpyDeviceToHost);
    cudaFree(gpu_points);
    cudaFree(gpu_buffer);
    cudaFree(gpu_rng_states);

}

Matrix<4, 4> create_random_affine_matrix_color() {
    Matrix<4, 4> scale_matrix;
    double data[4][4]= {
        {(double)stablerand_next(&stable_random) * 2 - 1, 0, 0, 0},
        {0, (double)stablerand_next(&stable_random) * 2 -1, 0, 0},
        {0, 0, (double)stablerand_next(&stable_random) * 2 - 1, 0},
        {0, 0, 0, 1}
    };

    /*double data[4][4]= {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };*/
    scale_matrix.setData(data);
    Matrix<4, 4> translation_matrix;
    double translation_data[4][4] = {
        {1, 0, 0, (double)stablerand_next(&stable_random) *2 - 1},
        {0, 1, 0, (double)stablerand_next(&stable_random) *2 - 1},
        {0, 0, 1, (double)stablerand_next(&stable_random) *2 - 1},
        {0, 0, 0, 1}
    };
    translation_matrix.setData(translation_data);

    double angleYaw = (double)stablerand_next(&stable_random) * 2 * OUR_PI - OUR_PI;
    double anglePitch = (double)stablerand_next(&stable_random) * 2 * OUR_PI - OUR_PI;
    double angleRoll = (double)stablerand_next(&stable_random) * 2 * OUR_PI - OUR_PI;

    double yaw_data[4][4] = {
        {cos(angleYaw), -sin(angleYaw), 0, 0},
        {sin(angleYaw), cos(angleYaw), 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };

    double pitch_data[4][4] = {
        {cos(anglePitch), 0, sin(anglePitch), 0},
        {0, 1, 0, 0},
        {-sin(anglePitch), 0, cos(anglePitch), 0},
        {0, 0, 0, 1}
    };

    double roll_data[4][4] = {
        {1, 0, 0, 0},
        {0, cos(angleRoll), -sin(angleRoll), 0},
        {0, sin(angleRoll), cos(angleRoll), 0},
        {0, 0, 0, 1}
    };
    Matrix<4, 4> yaw_matrix;
    yaw_matrix.setData(yaw_data);
    Matrix<4, 4> pitch_matrix;
    pitch_matrix.setData(pitch_data);
    Matrix<4, 4> roll_matrix;
    roll_matrix.setData(roll_data);

    Matrix<4, 4> rotation_matrix = yaw_matrix * pitch_matrix * roll_matrix;


    return scale_matrix * rotation_matrix * translation_matrix;
}

template <size_t T>
Matrix<T, T>* random_matrix_cpu(Matrix<T, T>* firstMatrix, Matrix<T, T>* secondMatrix, Matrix<T,T>* thridMatrix) {
    int index = ((int)(stablerand_next(&stable_random) * 3)) % 3;

    if (index == 0) {
        return firstMatrix;
    }
    if (index == 1) {
        return secondMatrix;
    }
    return thridMatrix;
}



void create_triangle_cpu_single(ColoredPoint* host_points, int amount, int iterations,
    int width, int height, char* output_buffer, float* cpu_measured_time, Matrix<3, 3>* pos_random_matrixes, Matrix<4, 4>* color_random_matrixes) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            for (int j = 0; j < amount; j++) {
                ColoredPoint* point = host_points + j;
                Matrix<3, 3>* random_pos_matrix = random_matrix_cpu(&pos_random_matrixes[0], &pos_random_matrixes[1], &pos_random_matrixes[2]);
                Matrix<4, 4>* random_color_matrix = random_matrix_cpu(&color_random_matrixes[0], &color_random_matrixes[1], &color_random_matrixes[2]);
                point->pos = (*random_pos_matrix) * point->pos;
                point->color = (*random_color_matrix) * point->color;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_ms = end - start;
        *cpu_measured_time = duration_ms.count();
}


int main() {
    // 4321
    stablerand_init(&stable_random, 929290504577);
    //srand(4321);
    int width = 1000;
    int height = 1000;
    int image_size = width * height;


    Matrix<3, 3> randomMatrixOne = create_random_affine_matrix();
    Matrix<3, 3> randomMatrixTwo = create_random_affine_matrix();
    Matrix<3, 3> randomMatrixThree = create_random_affine_matrix();
    Matrix<4, 4> randomColorMatrixOne = create_random_affine_matrix_color();
    Matrix<4, 4> randomColorMatrixTwo = create_random_affine_matrix_color();
    Matrix<4, 4> randomColorMatrixThree = create_random_affine_matrix_color();
    //randomMatrixOne.print();
    //randomColorMatrixOne.print();
    //randomMatrixTwo.print();
    //randomColorMatrixThree.print();

    Matrix<3, 3> matrixesArray[3] = {randomMatrixOne, randomMatrixTwo, randomMatrixThree};
    cudaMemcpyToSymbol(global_matrixes_data, &matrixesArray, sizeof(Matrix<3, 3>) * 3);

    Matrix<4,4> colorMatrixesArray[3] = { randomColorMatrixOne, randomColorMatrixTwo, randomColorMatrixThree};
    cudaMemcpyToSymbol(global_matrixes_data, &colorMatrixesArray, sizeof(Matrix<4, 4>) * 3, sizeof(Matrix<3, 3>) * 3);


    /*char bigTestBuffer[sizeof(Matrix<3, 3>) * 3 + sizeof(Matrix<4, 4>) * 3];
    memcpy(bigTestBuffer, matrixesArray, sizeof(sizeof(Matrix<3, 3>) * 3));

    memcpy(bigTestBuffer + sizeof(Matrix<3, 3>) * 3, colorMatrixesArray, sizeof(Matrix<4, 4>) * 3);
    char* rawPtrToColorPart = bigTestBuffer + sizeof(Matrix<3, 3>) * 3;
    Matrix<4, 4>* colTest = (Matrix<4, 4>*)rawPtrToColorPart;
    printf("printing color matrices\n");
    colTest[0].print();
    colTest[1].print();
    colTest[2].print();
    printf("printing color matrices done\n");*/



    // Generate random points
    int amount = 400000;
    //printf("Generating random points...\n");
    ColoredPoint* points = generate_random_points(amount);
    //points[0].color.print();

    ColoredPoint* buffer = (ColoredPoint*)malloc(amount * sizeof(ColoredPoint));
    char* outputBuffer = (char*)malloc(width * height * 3 * sizeof(char));

    //printf("done\n");
    printf("Creating triangle...\n");
    //create_triangle(points, amount, 200, buffer, &bottomLeftMatrix, &bottomRightMatrix, &topMatrix);
    // create_triangle_gpu(points, amount, 20, bottomLeftMatrix, bottomRightMatrix, topMatrix);
    create_triangle_gpu_with_frames(points, amount, 200, width, height);
    float unusedTime = 0;
    create_triangle_gpu_single(points, amount, 200, width, height, outputBuffer, &unusedTime);
    printf("Amount of iterations, Amount of points, Time GPU, Time CPU\n");

    /*for (int k = 10; k < 400; k += 20) {
        ColoredPoint* localPoints = generate_random_points(k);
        ColoredPoint* localBuffer = (ColoredPoint*)malloc(k * sizeof(ColoredPoint));
        ColoredPoint* clonedPointsForMeasureGPU = (ColoredPoint*) malloc(k * sizeof(ColoredPoint));
        ColoredPoint* clonedPointsForMeasureCPU = (ColoredPoint*) malloc(k * sizeof(ColoredPoint));

        for (int i = 10; i < 1000; i += 50) {

            float gpuTimeSum = 0;
            float cpuTimeSum = 0;
            for (int j = 0; j < 10; j++) {
                memcpy(clonedPointsForMeasureCPU, localPoints, k * sizeof(ColoredPoint));
                memcpy(clonedPointsForMeasureGPU, localPoints, k * sizeof(ColoredPoint));
                float gpuTime = 0;
                float cpuTime = 0;
                create_triangle_gpu_single(clonedPointsForMeasureCPU, k, i, width, height, outputBuffer, &gpuTime);
                create_triangle_cpu_single(clonedPointsForMeasureGPU, k, i, width, height, outputBuffer, &cpuTime, &matrixesArray[0], &colorMatrixesArray[0]);
                gpuTimeSum += gpuTime;
                cpuTimeSum += cpuTime;
            }
            gpuTimeSum /= 10;
            cpuTimeSum /= 10;
    
            printf("%d, %d, %f, %f\n", i, k, gpuTimeSum, cpuTimeSum);
    
        }
        free(localBuffer);
        free(clonedPointsForMeasureGPU);
        free(clonedPointsForMeasureCPU);
        free(localPoints);
    }*/

    save_image_with_name((uint8_t*) outputBuffer, width, height, "output.ppm");
    
    // Save the image
    //save_black_white_image(image_array, width, height);

    system("ffmpeg -y -framerate 5 -i ./vid_imgs/frame_%03d.ppm -c:v libx264 -pix_fmt yuv420p output.mp4");
    
    // Free the memory
    free(outputBuffer);
    free(points);
    free(buffer);
    
    return 0;
}