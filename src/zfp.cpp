#include <iostream>
#include <fstream>
#include <cmath>
#include <iostream>
#include <random>
#include <omp.h>
#include <random>
#include <iomanip>
#include <chrono>
#include <zfp.h>
#include <cfloat>
using namespace std;
// const int SIZE_X = 1024;
// const int SIZE_Y = 1024;
// const int SIZE_Z = 1024;
const int SIZE_X = 512;
const int SIZE_Y = 512;
const int SIZE_Z = 512;
const int BLK_SIZE = 6;
const int TOTAL_SIZE = SIZE_X * SIZE_Y * SIZE_Z;

typedef float* DataArray;

void saveBinaryData(const std::string& filePath, const DataArray& data) {
    std::ofstream outFile(filePath, std::ios::binary);

    if (!outFile.is_open()) {
        std::cerr << "Failed to open the output file for writing." << std::endl;
        delete[] data;
        return;
    }

    outFile.write(reinterpret_cast<const char*>(data), TOTAL_SIZE * sizeof(float));
    outFile.close();
}


float* ZFP_compress(void* array, size_t nx, size_t ny, size_t nz, double abs_error_bound, size_t* compressed_size) {
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    // Create metadata for 3D array
    zfp_type type = zfp_type_float;
    zfp_field* field = zfp_field_3d(array, type, nx, ny, nz);

    zfp_stream* zfp = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(zfp, abs_error_bound);

    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    void* buffer = malloc(bufsize);
    bitstream* stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    // Enable OpenMP if supported
    if (nx == SIZE_X) {
        if (!zfp_stream_set_execution(zfp, zfp_exec_omp)) {
            std::cout << "OpenMP not available" << std::endl;
        }
    }

    // Measure compression time
    size_t zfpsize = zfp_compress(zfp, field);
    auto t1 = high_resolution_clock::now();
    *compressed_size = zfpsize;
    double time_comp = duration_cast<duration<double>>(t1 - t0).count();
    std::cout << "[ZFP] Compression time: " << std::fixed << std::setprecision(5) << time_comp << " sec" << std::endl;


    // Measure decompression time
    t0 = high_resolution_clock::now();
    // Reset the stream and execution policy for decompression
    zfp_stream_rewind(zfp);
    zfp_stream_set_execution(zfp, zfp_exec_serial);  // or keep omp if desired

    float* decompressed_array = (float*)malloc(nx * ny * nz * sizeof(float));
    if (!decompressed_array) {
        std::cerr << "Memory allocation failed for decompressed data" << std::endl;
        return nullptr;
    }

    zfp_field_set_pointer(field, decompressed_array);

    if (!zfp_decompress(zfp, field)) {
        std::cerr << "Decompression failed" << std::endl;
        free(decompressed_array);
        decompressed_array = nullptr;
    }
    t1 = high_resolution_clock::now();
    double time_decomp = duration_cast<duration<double>>(t1 - t0).count();
    std::cout << "[ZFP] Decompression time: " << std::fixed << std::setprecision(5) << time_decomp << " sec" << std::endl;

    // Clean up
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
    free(buffer);

    return decompressed_array;
}



double computePSNR(const float* original, const float* decompressed, size_t num_elements) {
    double mse = 0.0;
    float max_val = -FLT_MAX;
    float min_val = FLT_MAX;

    for (size_t i = 0; i < num_elements; ++i) {
        float val = original[i];
        float diff = val - decompressed[i];
        mse += diff * diff;

        if (val > max_val) max_val = val;
        if (val < min_val) min_val = val;
    }

    mse /= static_cast<double>(num_elements);
    double range = static_cast<double>(max_val - min_val);

    if (mse == 0) return INFINITY;
    return 20.0 * log10(range) - 10.0 * log10(mse);
}


DataArray readBinaryData(const std::string& filePath) {
    DataArray data = new float[TOTAL_SIZE];
    std::ifstream inFile(filePath, std::ios::binary);

    inFile.read(reinterpret_cast<char*>(data), TOTAL_SIZE * sizeof(float));
    inFile.close();
    return data;
}



int main() {
    auto startRead= std::chrono::high_resolution_clock::now();
    // DataArray oriData = readBinaryData("/N/u/daocwang/BigRed200/data/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.f32");
    // DataArray oriData = readBinaryData("/N/u/daocwang/BigRed200/data/miranda_1024x1024x1024_float32.raw");
    DataArray oriData = readBinaryData("/N/u/daocwang/BigRed200/data/magnetic_reconnection_512x512x512_float32.raw");
    auto endRead = std::chrono::high_resolution_clock::now();
    double time_takenRead = std::chrono::duration_cast<std::chrono::nanoseconds>(endRead - startRead).count() * 1e-9; // Convert nanoseconds to seconds
    std::cout << "Time taken by Read is : "  << std::fixed << std::setprecision(5) << time_takenRead << " sec" << std::endl;

    double eb;
    std::ifstream eb_file("eb.txt");
    if (!eb_file)
    {
        std::cerr << "Error: Unable to open eb.txt" << std::endl;
        exit(1);
    }
    eb_file >> eb;

    auto startComp= std::chrono::high_resolution_clock::now();
    size_t compressed_size = 0;
    float* data = ZFP_compress(oriData, SIZE_X, SIZE_Y, SIZE_Z, eb, &compressed_size);
    auto endComp = std::chrono::high_resolution_clock::now();
    double time_takenComp = std::chrono::duration_cast<std::chrono::nanoseconds>(endComp - startComp).count() * 1e-9; // Convert nanoseconds to seconds
    // std::cout << "Time taken by Comp is : "  << std::fixed << std::setprecision(5) << time_takenComp << " sec" << std::endl;
    std::cout << "[ZFP] Compression Ratio: " << (TOTAL_SIZE * sizeof(float)) / (double)compressed_size << std::endl;
    double psnr = computePSNR(oriData, data, TOTAL_SIZE);
    std::cout << "[ZFP] PSNR: " << std::fixed << std::setprecision(3) << psnr  << std::endl;

    auto startWrite= std::chrono::high_resolution_clock::now();
    saveBinaryData("zfp.bin", data);
    auto endWrite = std::chrono::high_resolution_clock::now();
    double time_takenWrite = std::chrono::duration_cast<std::chrono::nanoseconds>(endWrite - startWrite).count() * 1e-9; 
    std::cout << "Time taken by Write is : "  << std::fixed << std::setprecision(5) << time_takenWrite << " sec" << std::endl;



    delete[] data;
    return 0;
}
