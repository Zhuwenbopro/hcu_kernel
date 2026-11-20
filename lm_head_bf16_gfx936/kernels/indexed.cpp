#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

#include <map>
#include <utility>

namespace {
using Key = std::pair<int, hipStream_t>;
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
};
thread_local ThreadHandles state;
}
extern "C" int head_indexed(int index, int m, int device, hipStream_t stream,
                            const void* x, const void* w, void* y) {
    const Key key(device,stream);
    auto it = state.handles.find(key);
    if (it == state.handles.end()) {
        rocblas_handle handle{};
        auto status=rocblas_create_handle(&handle);
        if(status != rocblas_status_success) return status;
        status=rocblas_set_stream(handle,stream);
        if(status != rocblas_status_success) {
            rocblas_destroy_handle(handle);
            return status;
        }
        it=state.handles.emplace(key,handle).first;
    }
    constexpr int n=124160, k=5120;
    constexpr auto bf=rocblas_datatype_bf16_r;
    const float alpha=1.0f, beta=0.0f;
    return rocblas_gemm_ex(it->second,rocblas_operation_none,rocblas_operation_none,
            n,m,k,&alpha,w,bf,n,x,bf,k,&beta,y,bf,n,y,bf,n,
            rocblas_datatype_f32_r,rocblas_gemm_algo_solution_index,index,0);
}
