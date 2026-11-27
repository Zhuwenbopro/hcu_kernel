// Each thread handles eight adjacent BF16 columns. The ten wave64 sums below
// occupy the first ten leaves in the unfused Triton operator's 16-wave tree.
constexpr int norm_threads = row_packs;
constexpr int wave_size = 64;
constexpr int row_waves = norm_threads / wave_size;
static_assert(norm_threads == 640 && row_waves == 10);

using FloatPair = float __attribute__((ext_vector_type(2)));
using PackedBits = uint32_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ PackedBits load_pack(const Pack* ptr) {
    return __builtin_bit_cast(PackedBits, *ptr);
}

__device__ __forceinline__ FloatPair unpack_bf16(uint32_t pair) {
    return FloatPair{__uint_as_float(pair << 16), __uint_as_float(pair & 0xffff0000u)};
}

__device__ __forceinline__ uint32_t rounded_bf16_bits(float value) {
    uint32_t bits = __float_as_uint(value);
    return __builtin_isnan(value) ? 0x7fff0000u : bits + 0x7fffu + ((bits >> 16) & 1u);
}

__device__ __forceinline__ uint32_t pack_bf16(FloatPair pair) {
    // Select the upper two bytes of each round-to-nearest-even FP32 value.
    return __builtin_amdgcn_perm(rounded_bf16_bits(pair.y),
                                 rounded_bf16_bits(pair.x), 0x07060302);
}

template<int control, int row_mask = 15>
__device__ __forceinline__ float dpp_value(float value) {
    int bits = __float_as_int(value);
    return __int_as_float(__builtin_amdgcn_update_dpp(bits, bits, control, row_mask, 15, true));
}

__device__ __forceinline__ float wave_sum(float value) {
    // The original reduction shifts within each 16-lane row by 8, 4, 2, 1,
    // then broadcasts lanes 15 and 31. Lane 63 holds the complete wave sum.
    value += dpp_value<280>(value);
    value += dpp_value<276>(value);
    value += dpp_value<274>(value);
    value += dpp_value<273>(value);
    value += dpp_value<322, 10>(value);
    value += dpp_value<323>(value);
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(value), 63));
}

__device__ __forceinline__ float row_scale(float* wave_sums, float epsilon) {
    __syncthreads();
    int lane = threadIdx.x % wave_size;
    if (threadIdx.x < wave_size) {
        float total = lane < row_waves ? wave_sums[lane] : 0.f;
        for (int step = 8; step; step >>= 1)
            total += __shfl_xor(total, step);
        if (lane == 0) {
            float width = float(hidden_size);
            // Match Triton's dynamic-width division and hardware sqrt, with
            // one rounded scale broadcast to every column including padding.
            asm("" : "+v"(width));
            float variance = __fdiv_rn(total, width) + epsilon;
            float root;
            asm("v_sqrt_f32 %0, %1" : "=v"(root) : "v"(variance));
            wave_sums[0] = __fdiv_rn(1.f, root);
        }
    }
    __syncthreads();
    return wave_sums[0];
}


__device__ __forceinline__ void finish_norm(PackedBits reduced, const Pack* residual,
    const void* weight, Pack* output, Pack* residual_output, int index,
    float* wave_sums, float epsilon) {
    const int pack = threadIdx.x;
    constexpr int bf16_weight = 1;
        PackedBits previous = load_pack(residual + index);
        PackedBits updated;
        float squares_sum = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            // Keep the BF16 cut after AllReduce. The FP32 sum supplies the
            // variance; its BF16-rounded value supplies the numerator.
            FloatPair added = unpack_bf16(reduced[j]) + unpack_bf16(previous[j]);
            updated[j] = pack_bf16(added);
            FloatPair squares = added * added;
            squares_sum += squares.x;
            squares_sum += squares.y;
        }
        residual_output[index] = __builtin_bit_cast(Pack, updated);
        float partial = wave_sum(squares_sum);
        if (pack % wave_size == 0)
            wave_sums[pack / wave_size] = partial;
        float scale = row_scale(wave_sums, epsilon);

        PackedBits weights, normalized_output;
        if (bf16_weight)
            weights = load_pack(static_cast<const Pack*>(weight) + pack);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            FloatPair raw_weight = bf16_weight ? unpack_bf16(weights[j])
                                               : static_cast<const FloatPair*>(weight)[pack * 4 + j];
            FloatPair normalized = unpack_bf16(updated[j]) * scale;
            normalized_output[j] = pack_bf16(normalized * (1.f + raw_weight));
        }
        output[index] = __builtin_bit_cast(Pack, normalized_output);
}
