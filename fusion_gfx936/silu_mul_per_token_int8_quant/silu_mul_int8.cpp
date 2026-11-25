#include "silu_mul_int8.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <iterator>

// Each backend owns its device LUT. The exported API initializes them together.
#define DECLARE_BACKEND(TAG) \
    extern "C" hipError_t launch_##TAG(int, int, int, int, const void*, void*, float*, hipStream_t); \
    extern "C" hipError_t init_##TAG(hipStream_t)

DECLARE_BACKEND(split8);
DECLARE_BACKEND(pair4);
DECLARE_BACKEND(pair8);
DECLARE_BACKEND(tp2_split8);
DECLARE_BACKEND(tp2_pair4);
#undef DECLARE_BACKEND

namespace {
enum class Layout { Split8, Pair4, Pair8 };
using Launcher = hipError_t (*)(int, const void*, int8_t*, float*, hipStream_t);

template <int D, Layout Kind, int Threads, int Schedule>
hipError_t launch(int rows, const void* input, int8_t* output, float* scales, hipStream_t stream) {
    if constexpr (D == 8704) {
        static_assert(Kind != Layout::Pair8);
        constexpr auto backend = Kind == Layout::Split8 ? launch_tp2_split8 : launch_tp2_pair4;
        return backend(D, Threads, Schedule, rows, input, output, scales, stream);
    } else {
        constexpr auto backend = Kind == Layout::Split8 ? launch_split8 :
                                 Kind == Layout::Pair4 ? launch_pair4 : launch_pair8;
        return backend(D, Threads, Schedule, rows, input, output, scales, stream);
    }
}

struct Route {
    int last_row_count;
    Launcher split;
    Launcher interleaved;
};

#include "routes.inc"

template <size_t N>
const Route& select_route(const Route (&routes)[N], int m) {
    return *std::lower_bound(std::begin(routes), std::end(routes), m,
        [](const Route& route, int rows) { return route.last_row_count < rows; });
}

hipError_t dispatch(bool interleaved, int m, int d, const void* input, int8_t* output,
                    float* scales, hipStream_t stream) {
    if (m < 0) return hipErrorInvalidValue;
    const Route* route;
    switch (d) {
        case 5120:  route = &select_route(routes5120, m); break;
        case 8704:  route = &select_route(routes8704, m); break;
        case 17408: route = &select_route(routes17408, m); break;
        default: return hipErrorInvalidValue;
    }
    if (m == 0) return hipSuccess;
    if (!input || !output || !scales || reinterpret_cast<uintptr_t>(input) % 16 ||
        reinterpret_cast<uintptr_t>(output) % 8 || reinterpret_cast<uintptr_t>(scales) % alignof(float)) {
        return hipErrorInvalidValue;
    }
    return (interleaved ? route->interleaved : route->split)(m, input, output, scales, stream);
}
} // namespace

extern "C" hipError_t silu_mul_int8_init(hipStream_t stream) {
    for (auto initialize : {init_split8, init_pair4, init_pair8, init_tp2_split8, init_tp2_pair4}) {
        const hipError_t status = initialize(stream);
        if (status != hipSuccess) return status;
    }
    return hipSuccess;
}

extern "C" hipError_t silu_mul_int8_split(int m, int d, const void* input, int8_t* output,
                                          float* scales, hipStream_t stream) {
    return dispatch(false, m, d, input, output, scales, stream);
}

extern "C" hipError_t silu_mul_int8_interleaved(int m, int d, const void* input, int8_t* output,
                                                float* scales, hipStream_t stream) {
    return dispatch(true, m, d, input, output, scales, stream);
}
