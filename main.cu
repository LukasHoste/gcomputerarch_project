#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include "matrix.h"
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>

#define OUR_PI 3.14159265358979323846
#define WIDTH 1920
#define HEIGHT 1920
#define CHANNELS 3

// Combined position and color point
struct ColoredPoint {
    Matrix<3, 1> pos;
    Matrix<3, 1> color;
};

template <typename T>
__host__ __device__ inline T my_clamp(const T& val, const T& lo, const T& hi) {
    return (val < lo) ? lo : (val > hi) ? hi : val;
}


__constant__ char global_matrices_data[6 * sizeof(Matrix<3, 3>)]; // 3 pos + 3 color matrices

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
        {(double)rand() / RAND_MAX, 0, 0},
        {0, (double)rand() / RAND_MAX, 0},
        {0, 0, 1},
    };
    matrix.setData(data);
    return matrix;
}

// https://lisyarus.github.io/blog/posts/transforming-colors-with-matrices.html
Matrix<3, 3> create_random_darkening_matrix() {
    Matrix<3, 3> matrix;
    double data[3][3] = {
        {1 - (double)rand() / RAND_MAX, 0, 0},
        {0, 1 - (double)rand() / RAND_MAX, 0},
        {0, 0, 1 - (double)rand() / RAND_MAX}
    };
    matrix.setData(data);
    return matrix;
}

// Generates a small random double between -maxChange and +maxChange
double random_small_change(double maxChange = 0.1) {
    return ((double)rand() / RAND_MAX) * 2 * maxChange - maxChange;
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
Matrix<3, 3> create_color_shift_matrix() {
    Matrix<3, 3> matrix;
    double data[3][3] = {
        {0, 0.5, 0},
        {0, 0, 0.5},
        {0.5, 0, 0}
    };
    matrix.setData(data);
    return matrix;
}


// lighten color
Matrix<3, 3> create_random_lighten_matrix() {
    Matrix<3, 3> matrix;
    double data[3][3] = {
        {1 + (double)rand() / RAND_MAX, 0, 0},
        {0, 1 + (double)rand() / RAND_MAX, 0},
        {0, 0, 1 + (double)rand() / RAND_MAX}
    };
    matrix.setData(data);
    return matrix;
}

// random tone matrix
// like this but fully random
// double col3[3][3] = {
    //     {0.393, 0.769, 0.189},
    //     {0.349, 0.686, 0.168},
    //     {0.272, 0.534, 0.131}
    // };
Matrix<3, 3> create_random_tone_matrix() {
    Matrix<3, 3> matrix;
    double data[3][3] = {
        {0.393 + (double)rand() / RAND_MAX, 0.769 + (double)rand() / RAND_MAX, 0.189 + (double)rand() / RAND_MAX},
        {0.349 + (double)rand() / RAND_MAX, 0.686 + (double)rand() / RAND_MAX, 0.168 + (double)rand() / RAND_MAX},
        {0.272 + (double)rand() / RAND_MAX, 0.534 + (double)rand() / RAND_MAX, 0.131 + (double)rand() / RAND_MAX}
    };
    matrix.setData(data);
    return matrix;
}

Matrix<3, 3> create_random_translation_matrix() {
    Matrix<3, 3> matrix;
    double data[3][3] = {
        {1, 0, (double)rand() / RAND_MAX},
        {0, 1, (double)rand() / RAND_MAX},
        {0, 0, 1},
    };
    matrix.setData(data);
    return matrix;
}

Matrix<3, 3> create_random_rotation_matrix() {
    Matrix<3, 3> matrix;
    double angle = ((double)rand() / RAND_MAX) * 2 * OUR_PI;
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
        {1, (double)rand() / RAND_MAX, 0},
        {(double)rand() / RAND_MAX, 1, 0},
        {0, 0, 1},
    };
    matrix.setData(data);
    return matrix;
}

Matrix<3, 3> create_random_affine_matrix() {
    return create_random_scale_matrix() * create_random_rotation_matrix()* create_random_translation_matrix();
}

Matrix<3, 3> create_random_affine_matrix_color() {
    return create_random_decrease_increase_matrix();
}

// Kernel for RNG setup
__global__ void setup_rng_kernel(curandState* states, int amount, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= amount) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// Main transformation kernel
__global__ void create_triangle_gpu_kernel(
    ColoredPoint* input, int amount, ColoredPoint* output,
    curandState* rng_states)
{
    Matrix<3, 3>* global_matrices = (Matrix<3, 3>*)global_matrices_data;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= amount) return;

    curandState* state = &rng_states[idx];
    float random = curand_uniform(state);
    
    int transform_index;
    if (random < 0.333f) transform_index = 0;
    else if (random < 0.666f) transform_index = 1;
    else transform_index = 2;

    Matrix<3, 3>* pos_matrix = &global_matrices[transform_index];
    Matrix<3, 3>* col_matrix = &global_matrices[transform_index + 3];

    ColoredPoint* current = &input[idx];
    ColoredPoint* next = &output[idx];

    // Transform position
    pos_matrix->mult(&current->pos, &next->pos);
    
    // Transform color (clamp after transformation)
    col_matrix->mult(&current->color, &next->color);
    
    // Keep alpha at 1.0
    next->color.at(2, 0) = 1.0;
    
    // Clamp color values between 0 and 1
    for(int i = 0; i < 2; i++) {
        next->color.at(i, 0) = fminf(fmaxf(next->color.at(i, 0), 0.0f), 1.0f);
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
            {(double)rand() / RAND_MAX},
            {(double)rand() / RAND_MAX},
            {1.0}
        };
        points[i].pos.setData(pos_data);
        
        // Start with white color
        double color_data[3][1] = {{1.0}, {1.0}, {1.0}};
        points[i].color.setData(color_data);
    }
    return points;
}

void rescale_points(ColoredPoint* points, int amount) {
    double min_x = 1, min_y = 1;
    double max_x = 0, max_y = 0;

    for (int i = 0; i < amount; i++) {
        double x = points[i].pos.at(0, 0);
        double y = points[i].pos.at(1, 0);
        
        min_x = fmin(min_x, x);
        min_y = fmin(min_y, y);
        max_x = fmax(max_x, x);
        max_y = fmax(max_y, y);
    }

    for (int i = 0; i < amount; i++) {
        points[i].pos.at(0, 0) = (points[i].pos.at(0, 0) - min_x) / (max_x - min_x);
        points[i].pos.at(1, 0) = (points[i].pos.at(1, 0) - min_y) / (max_y - min_y);
    }
}

uint8_t* scale_to_image(ColoredPoint* points, int amount) {
    float* color_accum = new float[WIDTH * HEIGHT * CHANNELS]{0};
    int* counts = new int[WIDTH * HEIGHT]{0};

    for (int i = 0; i < amount; i++) {
        int x = static_cast<int>(points[i].pos.at(0, 0) * (WIDTH - 1));
        int y = static_cast<int>((1.0 - points[i].pos.at(1, 0)) * (HEIGHT - 1));
        
        x = my_clamp(x, 0, WIDTH-1);
        y = my_clamp(y, 0, HEIGHT-1);

        int idx = y * WIDTH + x;
        counts[idx]++;
        
        color_accum[idx*3]   += points[i].color.at(0, 0);
        color_accum[idx*3+1] += points[i].color.at(1, 0);
        color_accum[idx*3+2] += points[i].color.at(2, 0);
    }

    uint8_t* image = new uint8_t[WIDTH * HEIGHT * CHANNELS];
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        if (counts[i] > 0) {
            image[i*3]   = static_cast<uint8_t>(255 * color_accum[i*3] / counts[i]);
            image[i*3+1] = static_cast<uint8_t>(255 * color_accum[i*3+1] / counts[i]);
            image[i*3+2] = static_cast<uint8_t>(255 * color_accum[i*3+2] / counts[i]);
        } else {
            image[i*3] = image[i*3+1] = image[i*3+2] = 0;
        }
    }

    delete[] color_accum;
    delete[] counts;
    return image;
}

void create_triangle_gpu(ColoredPoint* host_points, int amount, int iterations, int seed) {
    ColoredPoint *d_in, *d_out;
    curandState* d_states;

    cudaMalloc(&d_in, amount * sizeof(ColoredPoint));
    cudaMalloc(&d_out, amount * sizeof(ColoredPoint));
    cudaMalloc(&d_states, amount * sizeof(curandState));

    cudaMemcpy(d_in, host_points, amount * sizeof(ColoredPoint), cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (amount + threads - 1) / threads;
    
    setup_rng_kernel<<<blocks, threads>>>(d_states, amount, seed);
    cudaDeviceSynchronize();

    for (int i = 0; i < iterations; i++) {
        create_triangle_gpu_kernel<<<blocks, threads>>>(d_in, amount, d_out, d_states);
        std::swap(d_in, d_out);
    }

    cudaMemcpy(host_points, d_in, amount * sizeof(ColoredPoint), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_states);
}

int main() {
    // coole seeds: 2, 81, 4526
    unsigned int seed = 4526;
    srand(seed);
    const int NUM_POINTS = 10000000;
    const int ITERATIONS = 15;

    

    Matrix<3, 3> random_matrix_one = create_random_affine_matrix();
    Matrix<3, 3> random_matrix_two = create_random_affine_matrix();
    Matrix<3, 3> random_matrix_three = create_random_affine_matrix();

    // Define position transformations
    Matrix<3, 3> pos_matrices[3] = {
        random_matrix_one,
        random_matrix_two,
        random_matrix_three
    };

    // // Define position transformations
    // Matrix<3, 3> pos_matrices[3];
    
    // // Position matrix 1
    // double pos1[3][3] = {
    //     {0.5, 0, 0},
    //     {0, 0.5, 0},
    //     {0, 0, 1}
    // };
    // pos_matrices[0].setData(pos1);

    // // Position matrix 2
    // double pos2[3][3] = {
    //     {0.5, 0, 0.5},
    //     {0, 0.5, 0},
    //     {0, 0, 1}
    // };
    // pos_matrices[1].setData(pos2);

    // // Position matrix 3
    // double pos3[3][3] = {
    //     {0.5, 0, 0.25},
    //     {0, 0.5, 0.5},
    //     {0, 0, 1}
    // };
    // pos_matrices[2].setData(pos3);

    // Matrix<3, 3> random_color_matrix_one = create_random_affine_matrix_color();
    // Matrix<3, 3> random_color_matrix_two = create_random_affine_matrix_color();
    // Matrix<3, 3> random_color_matrix_three = create_random_affine_matrix_color();

    // // Define color transformations
    // Matrix<3, 3> col_matrices[3] = {
    //     random_color_matrix_one,
    //     random_color_matrix_two,
    //     random_color_matrix_three
    // };

    Matrix<3, 3> col_matrices[3];
    
    // Color matrix 1 (Increase red, decrease green)
    double col1[3][3] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
    col_matrices[0].setData(col1);

    col_matrices[1] = create_color_shift_matrix();

    // // Color matrix 3 (Sepia tone)
    // double col3[3][3] = {
    //     {0.393, 0.769, 0.189},
    //     {0.349, 0.686, 0.168},
    //     {0.272, 0.534, 0.131}
    // };
    // col_matrices[2].setData(col3);
    col_matrices[2] = create_random_affine_matrix_color();

    // Combine matrices for constant memory
    Matrix<3, 3> combined_matrices[6];
    for(int i = 0; i < 3; i++) {
        combined_matrices[i] = pos_matrices[i];
        combined_matrices[i+3] = col_matrices[i];
    }
    cudaMemcpyToSymbol(global_matrices_data, combined_matrices, sizeof(combined_matrices));

    // Generate points and process
    ColoredPoint* points = generate_random_points(NUM_POINTS);
    
    auto start = std::chrono::high_resolution_clock::now();
    create_triangle_gpu(points, NUM_POINTS, ITERATIONS, seed);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    printf("Processing time: %.2fs\n", elapsed.count());

    rescale_points(points, NUM_POINTS);
    uint8_t* image = scale_to_image(points, NUM_POINTS);
    save_image_array("output_image.ppm", image);

    delete[] points;
    delete[] image;
    return 0;
}