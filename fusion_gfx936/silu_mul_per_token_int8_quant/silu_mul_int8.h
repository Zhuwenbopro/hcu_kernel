#pragma once

#include <hip/hip_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the BF16 SiLU lookup table once per device on the given stream.
// Order the first use on every other stream after initialization.
hipError_t silu_mul_int8_init(hipStream_t stream);

// Contiguous BF16 input [m, 2*d], INT8 output [m, d], FP32 scales [m].
// Buffers must be disjoint and aligned to 16, 8 and 4 bytes respectively.
// Supported widths are 5120, 8704 and 17408. An empty batch is a no-op.
// In split layout each row contains d gates followed by d ups.
hipError_t silu_mul_int8_split(int m, int d, const void* input, int8_t* output,
                               float* scales, hipStream_t stream);

// Same dimensions and output order. In interleaved layout each row contains
// [gate_0, up_0, gate_1, up_1, ...].
hipError_t silu_mul_int8_interleaved(int m, int d, const void* input, int8_t* output,
                                     float* scales, hipStream_t stream);

#ifdef __cplusplus
}
#endif
