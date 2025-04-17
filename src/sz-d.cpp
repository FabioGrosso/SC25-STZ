#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <omp.h>
#include <iomanip>
#include <chrono>
#include "sz.hpp"
#include <cfloat>

using namespace std;

const int SIZE_X = 256;
const int SIZE_Y = 256;
const int SIZE_Z = 2048;
const int TOTAL_SIZE = SIZE_X * SIZE_Y * SIZE_Z;

typedef double* DataArray;

void saveBinaryData(const std::string& filePath, const DataArray& data) {
    std::ofstream outFile(filePath, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char*>(data), TOTAL_SIZE * sizeof(double));
    outFile.close();
}

DataArray readBinaryData(const std::string& filePath) {
    DataArray data = new double[TOTAL_SIZE];
    std::ifstream inFile(filePath, std::ios::binary);
    if (!inFile.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        delete[] data;
        return nullptr;
    }
    inFile.read(reinterpret_cast<char*>(data), TOTAL_SIZE * sizeof(double));
    inFile.close();
    return data;
}

double computePSNR(const double* original, const double* decompressed, size_t num_elements) {
    double mse = 0.0;
    double max_val = -FLT_MAX;
    double min_val = FLT_MAX;

    for (size_t i = 0; i < num_elements; ++i) {
        double val = original[i];
        double diff = val - decompressed[i];
        mse += diff * diff;

        if (val > max_val) max_val = val;
        if (val < min_val) min_val = val;
    }

    mse /= static_cast<double>(num_elements);
    double range = static_cast<double>(max_val - min_val);

    if (mse == 0) return INFINITY;
    return 20.0 * log10(range) - 10.0 * log10(mse);
}

char* AMR_compress(double* oriData, size_t blksize_x, size_t blksize_y, size_t blksize_z, double eb, size_t &outSize) {
    SZ3::Config conf(blksize_z, blksize_y, blksize_x);
    conf.cmprAlgo = SZ3::ALGO_INTERP;
    conf.lorenzo = true;
    conf.lorenzo2 = false;
    conf.regression = true;
    conf.regression2 = false;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = eb;
    conf.blockSize = 6;
    conf.openmp = true;
    char* compressedData = SZ_compress<double>(conf, oriData, outSize);
    return compressedData;
}

double* AMR_decompress(char* compressedData, size_t compressedSize, size_t blksize_x, size_t blksize_y, size_t blksize_z, double eb) {
    SZ3::Config conf(blksize_z, blksize_y, blksize_x);
    conf.cmprAlgo = SZ3::ALGO_INTERP;
    conf.lorenzo = true;
    conf.regression = true;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = eb;
    conf.blockSize = 6;
    conf.openmp = true;
    double* deData = new double[blksize_x * blksize_y * blksize_z];
    SZ_decompress<double>(conf, compressedData, compressedSize, deData);
    return deData;
}

int main() {
    auto t_start = std::chrono::high_resolution_clock::now();
    DataArray oriData = readBinaryData("/N/u/daocwang/BigRed200/data/wpx-1000-512.raw");
    auto t_end = std::chrono::high_resolution_clock::now();
    double read_time = std::chrono::duration<double>(t_end - t_start).count();
    cout << fixed << setprecision(5) << "Read time: " << read_time << " sec" << endl;

    double eb;
    std::ifstream eb_file("eb.txt");
    if (!eb_file)
    {
        std::cerr << "Error: Unable to open eb.txt" << std::endl;
        exit(1);
    }
    eb_file >> eb;
    size_t compressedSize = 0;
    t_start = std::chrono::high_resolution_clock::now();
    char* compressedData = AMR_compress(oriData, SIZE_X, SIZE_Y, SIZE_Z, eb, compressedSize);
    t_end = std::chrono::high_resolution_clock::now();
    double comp_time = std::chrono::duration<double>(t_end - t_start).count();
    cout << "Compression time: " << comp_time << " sec" << endl;

    cout << "Compression Ratio: " << (double)(TOTAL_SIZE * sizeof(double)) / compressedSize << endl;

    t_start = std::chrono::high_resolution_clock::now();
    double* decompressedData = AMR_decompress(compressedData, compressedSize, SIZE_X, SIZE_Y, SIZE_Z, eb);
    t_end = std::chrono::high_resolution_clock::now();
    double decomp_time = std::chrono::duration<double>(t_end - t_start).count();
    cout << "Decompression time: " << decomp_time << " sec" << endl;

    double psnr = computePSNR(oriData, decompressedData, TOTAL_SIZE);
    std::cout << "[SZ] PSNR: " << std::fixed << std::setprecision(3) << psnr  << std::endl;

    t_start = std::chrono::high_resolution_clock::now();
    saveBinaryData("sz_output.bin", decompressedData);
    t_end = std::chrono::high_resolution_clock::now();
    double write_time = std::chrono::duration<double>(t_end - t_start).count();
    cout << "Write time: " << write_time << " sec" << endl;

    delete[] oriData;
    delete[] compressedData;
    delete[] decompressedData;

    return 0;
}
