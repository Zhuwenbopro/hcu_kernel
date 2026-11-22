#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

#include <map>
#include <utility>

namespace {

using Key = std::pair<int, hipStream_t>;
constexpr int n = 124160, k = 5120;

struct ThreadHandles {
    std::map<Key, rocblas_handle> handles;

    ~ThreadHandles() {
        int previous = 0;
        if (hipGetDevice(&previous) != hipSuccess) return;
        for (const auto& item : handles) {
            if (hipSetDevice(item.first.first) == hipSuccess)
                rocblas_destroy_handle(item.second);
        }
        hipSetDevice(previous);
    }

    rocblas_status get(int device, hipStream_t stream, rocblas_handle& handle) {
        const Key key(device, stream);
        auto it = handles.find(key);
        if (it != handles.end()) {
            handle = it->second;
            return rocblas_status_success;
        }
        auto status = rocblas_create_handle(&handle);
        if (status != rocblas_status_success) return status;
        status = rocblas_set_stream(handle, stream);
        if (status != rocblas_status_success) {
            rocblas_destroy_handle(handle);
            return status;
        }
        handles.emplace(key, handle);
        return rocblas_status_success;
    }
};

thread_local ThreadHandles state;

int gemm(int index, int m, int width, rocblas_operation op_x, int pitch_x,
         int device, hipStream_t stream, const void* x, const void* w, void* y) {
    rocblas_handle handle{};
    auto status = state.get(device, stream, handle);
    if (status != rocblas_status_success) return status;

    constexpr auto bf = rocblas_datatype_bf16_r;
    const float alpha = 1.0f, beta = 0.0f;
    const auto algo = index ? rocblas_gemm_algo_solution_index : rocblas_gemm_algo_standard;
    return rocblas_gemm_ex(handle, rocblas_operation_none, op_x,
        width, m, k, &alpha, w, bf, n, x, bf, pitch_x, &beta, y, bf, n, y, bf, n,
        rocblas_datatype_f32_r, algo, index, 0);
}
}

extern "C" int head_indexed(int index, int m, int device, hipStream_t stream,
                            const void* x, const void* w, void* y) {
    return gemm(index, m, n, rocblas_operation_none, k, device, stream, x, w, y);
}

extern "C" int head_packed_x(int index, int m, int width, int device, hipStream_t stream,
                            const void* x, const void* w, void* y) {
    return gemm(index, m, width, rocblas_operation_transpose, m, device, stream, x, w, y);
}
