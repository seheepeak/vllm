/*
Adapted from https://github.com/turboderp/exllamav2 and https://github.com/qwopqwop200/GPTQ-for-LLaMa
*/

#include <cstdint>
#include <cstdio>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace vllm {
namespace gguf {

// from llama.cpp/ggml-cuda.cu
#define QK_K 256
#define K_SCALE_SIZE 12

enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 (5) support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    // k-quantizations
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
};

typedef struct
{
    half2 dm;                     // super-block scale for quantized scales/mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K / 8];         // quants, high bit
    uint8_t qs[QK_K / 2];         // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 176, "block_q5_K size is not 176");

typedef struct
{
    uint8_t ql[QK_K / 2];     // quants, lower 4 bits
    uint8_t qh[QK_K / 4];     // quants, upper 2 bits
    int8_t scales[QK_K / 16]; // scales
    half d;                   // delta
} block_q6_K;
static_assert(sizeof(block_q6_K) == 210, "block_q6_K size is not 210");

static inline __device__ void get_scale_min_k4(int j, const uint8_t *q, uint8_t &d, uint8_t &m)
{
    if (j < 4)
    {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    }
    else
    {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static __global__ void dequantize_block_q5_K(const void *__restrict__ vx, half *__restrict__ yy)
{
    const block_q5_K *x = (const block_q5_K *)vx;

    const int i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int tid = threadIdx.x;
    const int il = tid / 16; // il is in 0...3
    const int ir = tid % 16; // ir is in 0...15
    const int is = 2 * il;   // is is in 0...6

    half *y = yy + i * QK_K + 64 * il + 2 * ir;

    const float dall = __half2float(__low2half(x[i].dm));
    const float dmin = __half2float(__high2half(x[i].dm));

    const uint8_t *ql = x[i].qs + 32 * il + 2 * ir;
    const uint8_t *qh = x[i].qh + 2 * ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    uint8_t hm = 1 << (2 * il);
    y[0] = __float2half(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
    y[1] = __float2half(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = __float2half(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
    y[33] = __float2half(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
}

static __global__ void dequantize_block_q6_K(const void *__restrict__ vx, half *__restrict__ yy)
{
    const block_q6_K *x = (const block_q6_K *)vx;

    const int i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int tid = threadIdx.x;
    const int ip = tid / 32;      // ip is 0 or 1
    const int il = tid - 32 * ip; // 0...32
    const int is = 8 * ip + il / 16;

    half *y = yy + i * QK_K + 128 * ip + il;

    const float d = __half2float(x[i].d);

    const uint8_t *ql = x[i].ql + 64 * ip + il;
    const uint8_t qh = x[i].qh[32 * ip + il];
    const int8_t *sc = x[i].scales + is;

    y[0] = __float2half(d * sc[0] * ((int8_t)((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = __float2half(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = __float2half(d * sc[4] * ((int8_t)((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = __float2half(d * sc[6] * ((int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

static void dequantize_row_q5_K_cuda(const void *vx, half *y, const int k, cudaStream_t stream)
{
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

static void dequantize_row_q6_K_cuda(const void *vx, half *y, const int k, cudaStream_t stream)
{
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

using to_fp16_cuda_t = void (*)(const void *vx, half *y, const int k, cudaStream_t stream);
to_fp16_cuda_t ggml_get_to_fp16_cuda(int size_k, int quant_bytes_per_row)
{
    if (size_k % 256 != 0)
        throw std::runtime_error("size_k must be a multiple of 256");

    if (quant_bytes_per_row == size_k / 256 * 176)
        return dequantize_row_q5_K_cuda;
    else if (quant_bytes_per_row == size_k / 256 * 210)
        return dequantize_row_q6_K_cuda;
    else
        throw std::runtime_error("Unsupported weight_bytes_per_row");
}

void reconstruct_gemm
(
    cublasHandle_t cublas_handle,
    const half* a,
    const uint8_t* b_weight,
    half* c,
    half* temp_dq,
    int size_m,
    int size_n,
    int size_k,
    int quant_bytes_per_row
)
{
    const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(size_k, quant_bytes_per_row);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    to_fp16_cuda(b_weight, temp_dq, size_m * size_k, stream);

    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    cublasHgemm(cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                size_m, size_n, size_k,
                &alpha, temp_dq, size_k,
                        a,       size_k,
                &beta,  c,       size_m);
}

}  // namespace gguf
}  // namespace vllm

torch::Tensor gguf_gemm(torch::Tensor a, torch::Tensor b_weight)
{
    // q5_k => a: [n, k], b_weight: [m, k/256*176], c: [n, m]
    // q6_k => a: [n, k], b_weight: [m, k/256*210], c: [n, m]
    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
    at::Tensor c = torch::empty({a.size(0), b_weight.size(0)}, options);
    at::Tensor temp_dq = torch::empty({b_weight.size(0), a.size(1)}, options);

    vllm::gguf::reconstruct_gemm
    (
        at::cuda::getCurrentCUDABlasHandle(),
        (const half*) a.data_ptr(),
        (const uint8_t*) b_weight.data_ptr(),
        (half*) c.data_ptr(),
        (half*) temp_dq.data_ptr(),
        c.size(1),  // m
        c.size(0),  // n
        a.size(1),  // k
        b_weight.size(1) // quant_bytes_per_row
    );
    return c;
}
