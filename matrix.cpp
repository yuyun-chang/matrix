#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include "hip/hip_runtime.h"
using namespace std;

void print_mat(int row, int col, double* mat);
void fill_mat_vec(int ndl, int ndt, double* h_mat, double* h_vec, double* h_res);
void get_cpu_mat_vec_product(int ndl, int ndt, double* h_mat, double* h_vec, double* h_res);
void get_gpu_mat_vec_product(int ndl, int ndt, double* h_mat, double* h_vec, double* h_res, double* h_ans);
int get_block_size(int size);
bool check(int ndl, double* h_res, double* h_ans);
__global__ void mat_vec_product(int ndl, int ndt, double* d_mat, double* d_vec, double* d_res);

int main()
{
    int ndl = 5, ndt = 3;
    double* h_mat = (double*)malloc(sizeof(double) * ndl * ndt);
    double* h_vec = (double*)malloc(sizeof(double) * ndt);
    double* h_res = (double*)malloc(sizeof(double) * ndl);
    double* h_ans = (double*)malloc(sizeof(double) * ndl);
    fill_mat_vec(ndl, ndt, h_mat, h_vec, h_res);
    get_gpu_mat_vec_product(ndl, ndt, h_mat, h_vec, h_res, h_ans);
    get_cpu_mat_vec_product(ndl, ndt, h_mat, h_vec, h_res);
    printf("Check: %s\n", check(ndl, h_res, h_ans) ? "success" : "error");
    printf("\nCPU result:\n");
    print_mat(ndl, 1, h_res);
    printf("\nGPU result:\n");
    print_mat(ndl, 1, h_ans);
    return 0;
}

void print_mat(int row, int col, double* mat)
{
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%5.3f", mat[i*col + j]);
        }
        printf("\n");
    }
}

void fill_mat_vec(int ndl, int ndt, double* h_mat, double* h_vec, double* h_res)
{
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < ndl; i++) {
        for (int j = 0; j < ndt; j++) {
            h_mat[i*ndt + j] = dis(gen);
        }
    }
    for (int i = 0; i < ndt; i++) {
        h_vec[i] = dis(gen);
    }
    for (int i = 0; i < ndl; i++) {
        h_res[i] = dis(gen);
    }
}

void get_cpu_mat_vec_product(int ndl, int ndt, double* h_mat, double* h_vec, double* h_res)
{
    for (int i = 0; i < ndl; i++) {
        for (int j = 0; j < ndt; j++) {
            h_res[i] = h_res[i] + h_mat[i*ndt + j] * h_vec[j];
        }
    }
}

void get_gpu_mat_vec_product(int ndl, int ndt, double* h_mat, double* h_vec, double* h_res, double* h_ans)
{
    double* d_mat;
    double* d_vec;
    double* d_res;
    int block_sz;

    hipMalloc(&d_mat, sizeof(double) * ndl * ndt);
    hipMalloc(&d_vec, sizeof(double) * ndt);
    hipMalloc(&d_res, sizeof(double) * ndl);

    hipMemcpyHtoD(d_mat, h_mat, sizeof(double) * ndl * ndt);
    hipMemcpyHtoD(d_vec, h_vec, sizeof(double) * ndt);
    hipMemcpyHtoD(d_res, h_res, sizeof(double) * ndl);
    block_sz = get_block_size(ndt);

    hipLaunchKernelGGL(mat_vec_product, 1, block_sz, sizeof(double)*(ndt+block_sz), 0, ndl, ndt, d_mat, d_vec, d_res);

    hipMemcpyDtoH(h_ans, d_res, sizeof(double) * ndl);
}

int get_block_size(int size)
{
    if ((size & (size-1)) == 0) return size;
    int count = 0;
    while (size) {
        count++;
        size >>= 1;
    }
    return (1 << count);
}

bool check(int ndl, double* h_res, double* h_ans)
{
    for (int i = 0; i < ndl; i++) {
        if (abs(h_res[i] - h_ans[i]) >= 1e-6) {
            return false;
        }
    }
    return true;
}

__global__ void mat_vec_product(int ndl, int ndt, double* d_mat, double* d_vec, double* d_res)
{
    extern __shared__ double s_buffer[];
    double* s_vec = s_buffer;
    double* s_temp = &(s_vec[ndt]);
    int tid = threadIdx.x;
    int dim = blockDim.x;

    for (int i = tid; i < ndt; i += dim) {
        s_vec[i] = d_vec[i];
    }
    __syncthreads();

    for (int i = 0; i < ndl; i++) {
        s_temp[tid] = 0.0;
        for (int j = tid; j < ndt; j += dim) {
            s_temp[tid] = s_temp[tid] + d_mat[i*ndt + j] * s_vec[j];
        }
        __syncthreads();
        for (int offset = dim/2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                s_temp[tid] = s_temp[tid] + s_temp[tid + offset];
            }
            __syncthreads();
        }
        if (tid == 0) {
            d_res[i] = d_res[i] + s_temp[tid];
        }
        __syncthreads();
    }
}

