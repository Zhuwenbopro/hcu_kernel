struct alignas(8) Int8Pack {
    int8_t values[8];
};

constexpr float max_int8 = 127.f;
constexpr float min_absmax = 1.e-10f;

__device__ __forceinline__ FloatPair gemma_weight_pair(
    const void* weight, const PackedBits& bf16_weights, int bf16_weight, int pack, int pair) {
    const FloatPair raw = bf16_weight ? unpack_bf16(bf16_weights[pair])
                                      : static_cast<const FloatPair*>(weight)[pack * 4 + pair];
    return 1.f + raw;
}

__device__ __forceinline__ float pair_absmax(float maximum, FloatPair values) {
    return fmaxf(maximum, fmaxf(fabsf(values.x), fabsf(values.y)));
}

__device__ __forceinline__ void publish_wave_max(float* wave_maxima, float value) {
    value = fmaxf(value, dpp_value<280>(value));
    value = fmaxf(value, dpp_value<276>(value));
    value = fmaxf(value, dpp_value<274>(value));
    value = fmaxf(value, dpp_value<273>(value));
    value = fmaxf(value, dpp_value<322, 10>(value));
    value = fmaxf(value, dpp_value<323>(value));
    if (threadIdx.x % wave_size == wave_size - 1)
        wave_maxima[threadIdx.x / wave_size] = value;
}

template<bool Maximum>
__device__ __forceinline__ float reduce_row_partials(const float* partials) {
    const int lane = threadIdx.x % wave_size;
    float value = lane < row_waves ? partials[lane] : 0.f;
    // Every wave gets the statistic without a shared-memory broadcast barrier.
    for (int step = 8; step; step >>= 1) {
        const float peer = __shfl_xor(value, step);
        if constexpr (Maximum)
            value = fmaxf(value, peer);
        else
            value += peer;
    }
    return __shfl(value, 0, wave_size);
}

template<bool Dual, bool Fast>
__global__ __launch_bounds__(norm_threads) void fused_ar_int8(
    Peers peers, int rank, const Pack* input, const Pack* residual, const void* weight,
    Int8Pack* quantized_output, float* scales, Pack* residual_output, int rows,
    float epsilon, int bf16_weight) {
    const int pack = threadIdx.x;
    const uint32_t generation = next_generation(peers, rank);
    const size_t offset = Dual ? (generation & 1u) * peers.slot_packs : 0;
    Pack* local = peers.stage[rank] + offset;
    const Pack* remote = peers.stage[1 - rank] + offset;

    const int first = blockIdx.x * row_packs + pack;
    const PackedBits local_first = load_pack(input + first);
    local[first] = __builtin_bit_cast(Pack, local_first);
    for (int row = blockIdx.x + gridDim.x; row < rows; row += gridDim.x) {
        const int index = row * row_packs + pack;
        local[index] = input[index];
    }
    publish<Dual>(peers, rank, generation);

    __shared__ float wave_sums[row_waves];
    __shared__ float wave_maxima[row_waves];
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int index = row * row_packs + pack;
        const PackedBits a = row == blockIdx.x ? local_first : load_pack(input + index);
        const PackedBits reduced = allreduce_pairs(a, load_pack(remote + index));
        const PackedBits previous = load_pack(residual + index);
        PackedBits updated;
        PackedBits weights{};
        FloatPair weighted[4];
        float square_sum = 0.f;
        float local_max = 0.f;
        if constexpr (Fast) {
            if (bf16_weight)
                weights = load_pack(static_cast<const Pack*>(weight) + pack);
        }
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const FloatPair added = unpack_bf16(reduced[pair]) + unpack_bf16(previous[pair]);
            updated[pair] = pack_bf16(added);
            const FloatPair squares = added * added;
            square_sum += squares.x;
            square_sum += squares.y;
            if constexpr (Fast) {
                // Keep the residual sum in FP32 for the weighted numerator.
                weighted[pair] = added * gemma_weight_pair(weight, weights, bf16_weight, pack, pair);
                local_max = pair_absmax(local_max, weighted[pair]);
            }
        }
        residual_output[index] = __builtin_bit_cast(Pack, updated);
        const float partial = wave_sum(square_sum);
        if (pack % wave_size == 0)
            wave_sums[pack / wave_size] = partial;

        float rms_scale;
        float absmax;
        PackedBits normalized;
        if constexpr (Fast) {
            publish_wave_max(wave_maxima, local_max);
            __syncthreads();
            const float sum = reduce_row_partials<false>(wave_sums);
            rms_scale = rsqrtf(sum / float(hidden_size) + epsilon);
            absmax = reduce_row_partials<true>(wave_maxima) * rms_scale;
        } else {
            rms_scale = row_scale(wave_sums, epsilon);
            if (bf16_weight)
                weights = load_pack(static_cast<const Pack*>(weight) + pack);
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                // Match the original Gemma arithmetic order and quantize its BF16 result.
                const FloatPair value = unpack_bf16(updated[pair]) * rms_scale;
                normalized[pair] = pack_bf16(
                    value * gemma_weight_pair(weight, weights, bf16_weight, pack, pair));
                local_max = pair_absmax(local_max, unpack_bf16(normalized[pair]));
            }
            publish_wave_max(wave_maxima, local_max);
            __syncthreads();
            absmax = reduce_row_partials<true>(wave_maxima);
        }

        absmax = fmaxf(absmax, min_absmax);
        if (pack == 0)
            scales[row] = absmax / max_int8;
        const float inverse = max_int8 / absmax;
        Int8Pack output;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            FloatPair value;
            if constexpr (Fast)
                value = weighted[pair] * rms_scale;
            else
                value = unpack_bf16(normalized[pair]);
            output.values[pair * 2] = static_cast<int8_t>(__float2int_rz(nearbyintf(value.x * inverse)));
            output.values[pair * 2 + 1] = static_cast<int8_t>(__float2int_rz(nearbyintf(value.y * inverse)));
        }
        quantized_output[index] = output;
        if (row + gridDim.x < rows)
            __syncthreads();  // protect both arrays while the last waves read this row
    }
    if constexpr (!Dual)
        finish_single(peers, rank, generation);
}

template<bool Dual, bool Fast>
int int8_device_block_limit(int cus) {
    int resident_blocks;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident_blocks, fused_ar_int8<Dual, Fast>, norm_threads, 0) != hipSuccess)
        return 0;
    return resident_blocks * cus;
}

template<bool Dual>
void set_int8_block_limits(Context* context) {
    context->int8_caps[0] = int8_device_block_limit<Dual, false>(context->cus);
    context->int8_caps[1] = int8_device_block_limit<Dual, true>(context->cus);
}

template<bool Tunable>
int run_fused_int8(void* ctx, void* input, void* residual, void* weight, void* q,
                   float* scales, void* residual_output, int rows, int cols, int copyin,
                   float epsilon, void* stream, int selected_cap, int quant_mode) {
    auto* c = static_cast<Context*>(ctx);
    if (!c || (copyin != 1 && copyin != 2) || (quant_mode != 0 && quant_mode != 1) ||
        rows < 1 || rows > 4096 || cols != hidden_size ||
        !aligned(input) || !aligned(residual) || !aligned(weight) || !aligned(residual_output) ||
        !q || reinterpret_cast<uintptr_t>(q) % alignof(Int8Pack) != 0 ||
        !scales || reinterpret_cast<uintptr_t>(scales) % alignof(float) != 0 ||
        q == residual_output || q == scales || residual_output == scales ||
        rows * cols * sizeof(U16) > c->bytes)
        return -1;

    const int cap = selected_cap == 0 ? c->cap : selected_cap;
    if (cap < 1 || cap > c->cap || (!Tunable && cap != c->cap))
        return -1;
    const int blocks = std::min(cap, rows);
    if (blocks > c->int8_caps[quant_mode])
        return -1;

    const bool dual = c->peers.slot_packs != 0;
    const void* exact = dual ? reinterpret_cast<const void*>(fused_ar_int8<true, false>)
                             : reinterpret_cast<const void*>(fused_ar_int8<false, false>);
    const void* fast = dual ? reinterpret_cast<const void*>(fused_ar_int8<true, true>)
                            : reinterpret_cast<const void*>(fused_ar_int8<false, true>);
    int bf16_weight = copyin == 2;
    void* args[] = {&c->peers, &c->rank, &input, &residual, &weight, &q, &scales,
                    &residual_output, &rows, &epsilon, &bf16_weight};
    return static_cast<int>(hipLaunchKernel(quant_mode == 0 ? exact : fast, dim3(blocks),
        dim3(norm_threads), args, 0, static_cast<hipStream_t>(stream)));
}
