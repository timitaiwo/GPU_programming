// https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/

#define HIPBLAS_V2

#include <random>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


constexpr int MMA_M = 16, MMA_N = 16, MMA_K = 16;
constexpr int M0_M = 2, M0_N = 3, M0_K = 2;
constexpr int M1_M = M0_M, M1_N = 3, M1_K = M0_N;
constexpr int WMMA_DATA_WIDTH = 8;

using frag_type_f16 = _Float16 __attribute__((ext_vector_type(WMMA_DATA_WIDTH)));
using frag_type_f32 = float __attribute__((ext_vector_type( WMMA_DATA_WIDTH)));


__global__ void fused_gemm_TN(const __half* a0, const __half* b0, const __half* b1, float* c1) {
    const int lIdx = threadIdx.x;
    const int lane = lIdx % MMA_K;
    const int laneGroup = lIdx / MMA_K;

    frag_type_f16 a0_frag[M0_M][M0_K];
    frag_type_f16 b0_frag[M0_N][M0_K];
    frag_type_f32 c0_frag[M0_M][M0_N] = {};

    constexpr int m0_stride_m = MMA_M * M0_K * MMA_K;
    constexpr int m0_stride_n = MMA_N * M0_K * MMA_K;
    constexpr int m0_stride_k = MMA_K;
    constexpr int m0_stride_mma_m = M0_K * MMA_K;
    constexpr int m0_stride_mma_n = M0_K * MMA_K;

    for(int m = 0; m < M0_M; ++m) {
        for(int k = 0; k < M0_K; ++k) {
            int block_idx = m * m0_stride_m + k * m0_stride_k;
            int lane_idx = lane * m0_stride_mma_m;
            int lane_group_idx = laneGroup * WMMA_DATA_WIDTH;
            a0_frag[m][k] = reinterpret_cast<const frag_type_f16&>(a0[block_idx + lane_idx + lane_group_idx]);
        }
    }

    for(int n = 0; n < M0_N; ++n) {
        for(int k = 0; k < M0_K; ++k) {
            int block_idx = n * m0_stride_n + k * m0_stride_k;
            int lane_idx = lane * m0_stride_mma_n;
            int lane_group_idx = laneGroup * WMMA_DATA_WIDTH;
            b0_frag[n][k] = reinterpret_cast<const frag_type_f16&>(b0[block_idx + lane_idx + lane_group_idx]);
        }
    }

    for(int m = 0; m < M0_M; ++m) {
        for(int n = 0; n < M0_N; ++n) {
            for(int k = 0; k < M0_K; ++k) {
                c0_frag[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0_frag[n][k], a0_frag[m][k], c0_frag[m][n]);
            }
        }
    }

    static_assert(M0_M == M1_M, "M0's M must equal to M1's M");
    static_assert(M0_N * MMA_N == M1_K * MMA_K, "M0's N must equal to M1's K");

    frag_type_f16 a1_frag[M1_M][M1_K];

    for(int m = 0; m < M1_M; ++m) {
        for(int k = 0; k < M1_K; ++k) {
            for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                a1_frag[m][k][ele] = __float2half(c0_frag[m][k][ele]);
            }
        }
    }

    frag_type_f16 b1_frag[M1_N][M1_K];
    frag_type_f32 c1_frag[M1_M][M1_N] = {};

    constexpr int m1_stride_n = MMA_N * M1_K * MMA_K;
    constexpr int m1_stride_k = MMA_K;
    constexpr int m1_stride_mma_n = M1_K * MMA_K;

    for(int n = 0; n < M1_N; ++n) {
        for(int k = 0; k < M1_K; ++k) {
            int block_idx = n * m1_stride_n + k * m1_stride_k;
            int lane_idx = lane * m1_stride_mma_n;
            int lane_group_idx = laneGroup * WMMA_DATA_WIDTH;
            b1_frag[n][k] = reinterpret_cast<const frag_type_f16&>(b1[block_idx + lane_idx + lane_group_idx]);
        }
    }

    for(int m = 0; m < M1_M; ++m) {
        for(int n = 0; n < M1_N; ++n) {
            for(int k = 0; k < M1_K; ++k) {
                c1_frag[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1_frag[n][k], a1_frag[m][k], c1_frag[m][n]);
            }
        }
    }

    constexpr int c1_stride_m = MMA_M * M1_N * MMA_N;
    constexpr int c1_stride_n = MMA_N;
    constexpr int c1_stride_mma_n = M1_N * MMA_N;

    for(int m = 0; m < M1_M; ++m) {
        for(int n = 0; n < M1_N; ++n) {
            int block_idx = m * c1_stride_m + n * c1_stride_n;
            int lane_idx = lane * c1_stride_mma_n;
            int lane_group_idx = laneGroup * WMMA_DATA_WIDTH;
            reinterpret_cast<frag_type_f32&>(c1[block_idx + lane_idx + lane_group_idx]) = c1_frag[m][n];
        }
    }
}

template <typename T, typename RNG>
void gen_rand_data(T* data, size_t n, RNG &rng) {
    std::normal_distribution<float> nd(-100, 100);
    for (size_t i = 0; i < n; ++i) {
        float v = nd(rng) * 0.01f;
        data[i] = v;
    }
}

int main(int argc, char* argv[]) {
    static_assert(MMA_M == 16, "MMA_M must be 16 for GFX12");
    static_assert(MMA_N == 16, "MMA_N must be 16 for GFX12");
    static_assert(MMA_K == 16, "MMA_K must be 16 for GFX12");
    static_assert(WMMA_DATA_WIDTH == 8, "WMMA_DATA_WIDTH must be 8 for GFX12");

    int M0 = M0_M * MMA_M, N0 = M0_N * MMA_N, K0 = M0_K * MMA_K;
    int M1 = M0, N1 = M1_N * MMA_N, K1 = N0;

    thrust::host_vector<__half> h_A0(M0 * K0);
    thrust::host_vector<__half> h_B0(N0 * K0);
    thrust::host_vector<__half> h_B1(N1 * K1);
    thrust::host_vector<float> h_C1(M1 * N1);

    std::mt19937 rng(2025);
    gen_rand_data(h_A0.data(), h_A0.size(), rng);
    gen_rand_data(h_B0.data(), h_B0.size(), rng);
    gen_rand_data(h_B1.data(), h_B1.size(), rng);

    thrust::device_vector<__half> d_A0 = h_A0;
    thrust::device_vector<__half> d_B0 = h_B0;
    thrust::device_vector<__half> d_B1 = h_B1;
    thrust::device_vector<float> d_C1 = h_C1;

    fused_gemm_TN<<<dim3(1), dim3(32, 1, 1), 0, 0>>>(d_A0.data().get(), d_B0.data().get(), d_B1.data().get(), d_C1.data().get());

    auto err = hipDeviceSynchronize();
    printf("err = %d, str = %s\n", err, hipGetErrorString(err));
    err = hipGetLastError();
    printf("err = %d, str = %s\n", err, hipGetErrorString(err));

    h_C1 = d_C1;

    thrust::device_vector<__half> d_C0_blas(M0 * N0);
    thrust::device_vector<float> d_C1_blas(M1 * N1);

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    float alpha(1.0f);
    float beta(0.0f);

    hipblasStatus_t ret = hipblasGemmEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N, N0, M0, K0,
        &alpha, d_B0.data().get(), HIP_R_16F, K0,
        d_A0.data().get(), HIP_R_16F, K0, &beta,
        d_C0_blas.data().get(), HIP_R_16F, N0,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);

    if (ret == HIPBLAS_STATUS_SUCCESS) {
        ret = hipblasGemmEx(
            handle, HIPBLAS_OP_T, HIPBLAS_OP_N, N1, M1, K1,
            &alpha, d_B1.data().get(), HIP_R_16F, K1,
            d_C0_blas.data().get(), HIP_R_16F, K1, &beta,
            d_C1_blas.data().get(), HIP_R_32F, N1,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);

            if (ret == HIPBLAS_STATUS_SUCCESS) {
                constexpr float threshold = 0.001;
                int diff_num = 0;
                thrust::host_vector<float> h_C1_blas = d_C1_blas;
                for(int i = 0; i < M1 * N1; ++i) {
                    float diff = std::abs(h_C1[i] - h_C1_blas[i]);
                    if(diff > threshold) {
                        printf("%f %f\n", h_C1[i], h_C1_blas[i]);
                        diff_num++;
                    }
                }

                if(!diff_num) {
                    printf("Fused GEMM has same result as two BLAS GEMM.\n");
                }
            } else {
                printf("hipblas err = %d, str = %s\n", ret, hipblasStatusToString(ret));
            }
    } else {
        printf("hipblas err = %d, str = %s\n", ret, hipblasStatusToString(ret));
    }

    hipblasDestroy(handle);

    return 0;
}