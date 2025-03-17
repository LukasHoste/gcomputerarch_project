#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

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

int main() {
    // Define the image dimensions
    int width = 1920;
    int height = 1080;
    int image_size = width * height;
    
    // Allocate memory for the image
    uint8_t* image_array = (uint8_t*)malloc(image_size);
    
    // Fill the image with random data
    for (int i = 0; i < image_size; i++) {
        image_array[i] = rand() % 256;
    }
    
    // Save the image
    save_black_white_image(image_array, width, height);
    
    // Free the memory
    free(image_array);
    
    return 0;
}