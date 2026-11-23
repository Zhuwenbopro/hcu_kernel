#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

extern "C" int gate_up_blas_init(rocblas_handle* handle) {
    return rocblas_create_handle(handle);
}

extern "C" int gate_up_blas_destroy(rocblas_handle handle) {
    return rocblas_destroy_handle(handle);
}

extern "C" int gate_up_kn_gemm(rocblas_handle handle, hipStream_t stream,
                               const void* x, const void* w, void* y,
                               int m, int width, int k, int n, int pitch_x, int trans_x) {
    auto status = rocblas_set_stream(handle, stream);
    if (status != rocblas_status_success) return status;

    constexpr auto bf16 = rocblas_datatype_bf16_r;
    const float one = 1.0f, zero = 0.0f;
    auto op_x = trans_x ? rocblas_operation_transpose : rocblas_operation_none;
    // Column-major Y^T = W^T X^T. W and Y slices retain the full shard stride.
    return rocblas_gemm_ex(handle, rocblas_operation_none, op_x, width, m, k,
                           &one, w, bf16, n, x, bf16, pitch_x, &zero,
                           y, bf16, n, y, bf16, n, rocblas_datatype_f32_r,
                           rocblas_gemm_algo_standard, 0, 0);
}
