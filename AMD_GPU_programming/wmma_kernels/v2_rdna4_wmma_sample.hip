// https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/

#define HIPBLAS_V2

#include <random>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

constexpr int MMA_M = 16, MMA_N = 16, MMA_K = 32;
constexpr int M0_M = 2, M0_N = 3, M0_K = 2;
constexpr int WMMA_DATA_WIDTH = 8;

using frag_i8_16 = int8_t __attribute__((ext_vector_type(WMMA_DATA_WIDTH * 2)));
using frag_i8_8 = int8_t __attribute__((ext_vector_type(WMMA_DATA_WIDTH)));
using frag_i32_8 = int32_t __attribute__((ext_vector_type(WMMA_DATA_WIDTH)));


template <bool signed_a = true, bool signed_b = true, bool signed_c = true>
__forceinline__ __device__ void mma_i32i8_16_16_32_gfx12(const frag_i8_16& am, const frag_i8_16& bm, frag_i32_8& cm) {
    const int8_t* a_ptr = reinterpret_cast<const int8_t*>(&am);
    const int8_t* b_ptr = reinterpret_cast<const int8_t*>(&bm);

    cm = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
        signed_a, reinterpret_cast<const frag_i8_8&>(a_ptr[0]),
        signed_b, reinterpret_cast<const frag_i8_8&>(b_ptr[0]),
        cm, signed_c);

    cm = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
        signed_a, reinterpret_cast<const frag_i8_8&>(a_ptr[WMMA_DATA_WIDTH]),
        signed_b, reinterpret_cast<const frag_i8_8&>(b_ptr[WMMA_DATA_WIDTH]),
        cm, signed_c);
}

__global__ void fused_mma_TN(const int8_t* a, const int8_t* b, int32_t* c) {
    constexpr int ab_size = sizeof(frag_i8_16) / sizeof(int8_t);
    constexpr int c_size = sizeof(frag_i32_8) / sizeof(int32_t);

    const int lIdx = threadIdx.x;
    const int lane = lIdx % ab_size;
    const int lane_group = lIdx / ab_size;

    frag_i8_16 a_frag[M0_M][M0_K];
    frag_i8_16 b_frag[M0_N][M0_K];
    frag_i32_8 c_frag[M0_M][M0_N] = {};

    constexpr int m0_stride_m = MMA_M * M0_K * MMA_K;
    constexpr int m0_stride_n = MMA_N * M0_K * MMA_K;
    constexpr int m0_stride_k = MMA_K;
    constexpr int m0_stride_mma_m = M0_K * MMA_K;
    constexpr int m0_stride_mma_n = M0_K * MMA_K;

    for (int m = 0; m < M0_M; ++m) {
        for (int k = 0; k < M0_K; ++k) {
            int block_idx = m * m0_stride_m + k * m0_stride_k;
            int lane_idx = lane * m0_stride_mma_m;
            int lane_group_idx = lane_group * ab_size;
            a_frag[m][k] = reinterpret_cast<const frag_i8_16&>(a[block_idx + lane_idx + lane_group_idx]);
        }
    }

    for (int n = 0; n < M0_N; ++n) {
        for (int k = 0; k < M0_K; ++k) {
            int block_idx = n * m0_stride_n + k * m0_stride_k;
            int lane_idx = lane * m0_stride_mma_n;
            int lane_group_idx = lane_group * ab_size;
            b_frag[n][k] = reinterpret_cast<const frag_i8_16&>(b[block_idx + lane_idx + lane_group_idx]);
        }
    }

    for (int m = 0; m < M0_M; ++m) {
        for (int n = 0; n < M0_N; ++n) {
            for (int k = 0; k < M0_K; ++k) {
                mma_i32i8_16_16_32_gfx12<true, true, true>(a_frag[m][k], b_frag[n][k], c_frag[m][n]);
            }
        }
    }

    constexpr int c_stride_m = MMA_M;
    constexpr int c_stride_n = MMA_N * M0_M * MMA_M;
    constexpr int c_stride_mma_n = M0_M * MMA_M;

    for (int n = 0; n < M0_N; ++n) {
        for (int m = 0; m < M0_M; ++m) {
            int block_idx = m * c_stride_m + n * c_stride_n;
            int lane_idx = lane * c_stride_mma_n;
            int lane_group_idx = lane_group * c_size;
            reinterpret_cast<frag_i32_8&>(c[block_idx + lane_idx + lane_group_idx]) = c_frag[m][n];
        }
    }
}

template <typename T, typename RNG>
void gen_rand_data(T* data, size_t n, RNG& rng) {
    std::uniform_int_distribution<int32_t> nd(-3, 3);
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<T>(nd(rng));
    }
}

int main(int argc, char** argv) {
    static_assert(MMA_M == 16, "MMA_M must be 16 for GFX12");
    static_assert(MMA_N == 16, "MMA_N must be 16 for GFX12");
    static_assert(MMA_K == 32, "MMA_K must be 16 for wide GFX12");
    static_assert(WMMA_DATA_WIDTH == 8, "WMMA_DATA_WIDTH must be 8 for GFX12");

    int M0 = M0_M * MMA_M, N0 = M0_N * MMA_N, K0 = M0_K * MMA_K;

    thrust::host_vector<int8_t> h_A(M0 * K0);
    thrust::host_vector<int8_t> h_B(N0 * K0);
    thrust::host_vector<int32_t> h_C(M0 * N0);

    std::mt19937 rng(2025);
    gen_rand_data(h_A.data(), h_A.size(), rng);
    gen_rand_data(h_B.data(), h_B.size(), rng);

    thrust::device_vector<int8_t> d_A = h_A;
    thrust::device_vector<int8_t> d_B = h_B;
    thrust::device_vector<int32_t> d_C = h_C;

    fused_mma_TN<<<dim3(1), dim3(32, 1, 1), 0, 0>>>(d_A.data().get(), d_B.data().get(), d_C.data().get());

    auto err = hipDeviceSynchronize();
    printf("err = %d, str = %s\n", err, hipGetErrorString(err));
    err = hipGetLastError();
    printf("err = %d, str = %s\n", err, hipGetErrorString(err));

    h_C = d_C;

    thrust::device_vector<int32_t> d_C_blas(M0 * N0, 0);

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    int32_t alpha(1);
    int32_t beta(0);

    hipblasStatus_t ret = hipblasGemmEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N, M0, N0, K0,
        &alpha, d_A.data().get(), HIP_R_8I, K0,
        d_B.data().get(), HIP_R_8I, K0, &beta,
        d_C_blas.data().get(), HIP_R_32I, M0,
        HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT);

    if (ret == HIPBLAS_STATUS_SUCCESS) {
        int diff_num = 0;
        thrust::host_vector<int32_t> h_C_blas = d_C_blas;

        for (int i = 0; i < M0 * N0; ++i) {
            if (h_C[i] != h_C_blas[i]) {
                printf("%i %i\n", h_C[i], h_C_blas[i]);
                diff_num++;
            }
        }

        if (!diff_num) {
            printf("Wide MMA has same result as MMA in BLAS GEMM.\n");
        }
    } else {
        printf("hipblas err = %d, str = %s\n", ret, hipblasStatusToString(ret));
    }

    hipblasDestroy(handle);

    return 0;
}