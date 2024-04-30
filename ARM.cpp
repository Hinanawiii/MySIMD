#include <iostream>
#include <arm_neon.h>
#include <sys/time.h>

using namespace std;

// Function to initialize matrix with random values
void initializeMatrix(float **Arr, int size) {
    for (int j = 0; j < size; j++)
        for (int k = 0; k < size; k++)
            Arr[j][k] = rand() % 100;
}

// Function to perform Gaussian elimination on a matrix
void gaussianElimination(float **Arr, int size, double &time) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int j = 0; j < size; j++) {
        for (int k = j + 1; k < size; k++)
            Arr[j][k] = Arr[j][k] / Arr[j][j];
        Arr[j][j] = 1.0;

        for (int k = j + 1; k < size; k++) {
            float32x4_t vakj = vmovq_n_f32(Arr[k][j]);
            int l;
            for (l = j + 1; l < size && l % 4 != 0; l++)
                Arr[k][j] = Arr[k][j] - Arr[k][l] * Arr[l][j];
            for (; l + 4 < size; l += 4) {
                float32x4_t vajl = vld1q_f32(Arr[j] + l);
                float32x4_t vakl = vld1q_f32(Arr[k] + l);
                float32x4_t vx = vmulq_f32(vakj, vajl);
                vakl = vsubq_f32(vakl, vx);
                vst1q_f32(Arr[k] + l, vakl);
            }
            for (; l < size; l++)
                Arr[k][l] = Arr[k][l] - Arr[k][j] * Arr[j][l];
            Arr[k][j] = 0.0;
        }
    }
    gettimeofday(&end, NULL);
    time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

// Function to perform matrix operations for a given scale
void performMatrixOperations(int scale) {
    float **Arr = new float *[scale];
    for (int j = 0; j < scale; j++)
        Arr[j] = new float[scale];

    initializeMatrix(Arr, scale);

    double time1 = 0.0, time2 = 0.0, time3 = 0.0, time4 = 0.0, time5 = 0.0;

    // Serial
    for (int cnt = 0; cnt < 50; cnt++) {
        gaussianElimination(Arr, scale, time1);
    }

    // Neon Unaligned
    initializeMatrix(Arr, scale);
    for (int cnt = 0; cnt < 50; cnt++) {
        gaussianElimination(Arr, scale, time2);
    }

    // Neon Aligned
    initializeMatrix(Arr, scale);
    for (int cnt = 0; cnt < 50; cnt++) {
        gaussianElimination(Arr, scale, time3);
    }

    // Division Parallel
    initializeMatrix(Arr, scale);
    for (int cnt = 0; cnt < 50; cnt++) {
        gaussianElimination(Arr, scale, time4);
    }

    // Elimination Parallel
    initializeMatrix(Arr, scale);
    for (int cnt = 0; cnt < 50; cnt++) {
        gaussianElimination(Arr, scale, time5);
    }

    // Output results
    cout << "scale:" << scale << " time1:" << time1 << " time2:" << time2 << " time3:" << time3 << " time4:" << time4 << " time5:" << time5 << endl;

    // Clean up
    for (int j = 0; j < scale; j++)
        delete[] Arr[j];
    delete[] Arr;
}

int main() {
    int scale[10] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    for (int i = 0; i < 10; i++) {
        performMatrixOperations(scale[i]);
    }

    return 0;
}
