// https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-3/

#define WMMA_DATA_WIDTH 8
typedef _Float16 frag_type __attribute__((ext_vector_type(WMMA_DATA_WIDTH)));

__device__ __forceinline__ frag_type shfl_movmatrix(const frag_type& t) {
    frag_type ret;
    auto lane_id = __lane_id();
    auto v_src = lane_id % 8;

    auto TLayout = [] (auto lane_id) {
        constexpr unsigned shape[] = {8, 2, 2};
        constexpr unsigned stride[] = {0, 16, 8};

        unsigned result = 0;
        for(int i = 0; i < sizeof(shape) / sizeof(shape[0]); ++i) {
            result += lane_id % shape[i] * stride[i];
            lane_id /= shape[i];
        }
        return result;
    };

    auto t_trans = TLayout(lane_id);

#pragma unroll
    for(int v = 0; v < 8; ++v) {
        auto t_src = t_trans + v;

        frag_type reg;
        uint32_t* in_reg = (uint32_t*)(&t);
        uint32_t* out_reg = (uint32_t*)(&reg);

        static_assert(sizeof(reg) % sizeof(*in_reg) == 0, "frag_type must be dividend by uint32_t evenly");

#pragma unroll
        for(int tv = 0; tv < sizeof(reg) / sizeof(*in_reg); ++tv) {
            out_reg[tv] = __shfl(in_reg[tv], t_src);
        }

        ret[v] = reg[v_src];
    }
    return ret;
}