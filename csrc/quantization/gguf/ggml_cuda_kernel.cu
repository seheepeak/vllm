#include <cstdint>
#include <cstdio>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace vllm {
namespace gguf {

// from llama.cpp/ggml-cuda.cu
#define QK_K 256
#define K_SCALE_SIZE 12
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
#define WARP_SIZE 32
#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
#define MMVQ_MAX_BATCH_SIZE 8 // max batch size to use MMVQ kernels
#define MMQ_MAX_BATCH_SIZE 32 // max batch size to use MMQ kernels when tensor cores are available

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q6_K_Q8_1_MMVQ 1

#define GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            fflush(stdout); \
            fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

[[noreturn]] static void ggml_cuda_error(const char *stmt, const char *func, const char *file, const int line, const char *msg)
{
    int id = -1; // in case cudaGetDevice fails
    cudaGetDevice(&id);

    fprintf(stderr, "CUDA error: %s\n", msg);
    fprintf(stderr, "  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    fprintf(stderr, "  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ASSERT(!"CUDA error");
}

#define CUDA_CHECK_GEN(err, success, error_fn)                                   \
    do                                                                           \
    {                                                                            \
        auto err_ = (err);                                                       \
        if (err_ != (success))                                                   \
        {                                                                        \
            ggml_cuda_error(#err, __func__, __FILE__, __LINE__, error_fn(err_)); \
        }                                                                        \
    } while (0)

#define CUDA_CHECK(err) CUDA_CHECK_GEN(err, cudaSuccess, cudaGetErrorString)


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

#define QR5_K 2
#define QI5_K (QK_K / (4 * QR5_K))
typedef struct
{
    half2 dm;                     // super-block scale for quantized scales/mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K / 8];         // quants, high bit
    uint8_t qs[QK_K / 2];         // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 176, "block_q5_K size is not 176");

#define QR6_K 2
#define QI6_K (QK_K / (4 * QR6_K))
typedef struct
{
    uint8_t ql[QK_K / 2];     // quants, lower 4 bits
    uint8_t qh[QK_K / 4];     // quants, upper 2 bits
    int8_t scales[QK_K / 16]; // scales
    half d;                   // delta
} block_q6_K;
static_assert(sizeof(block_q6_K) == 210, "block_q6_K size is not 210");

#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
typedef struct
{
    half2 ds;         // ds.x = delta, ds.y = sum
    int8_t qs[QK8_1]; // quants
} block_q8_1;

template <typename dst_t>
__device__ dst_t convert_type(float value);

template <>
__device__ float convert_type<float>(float value) {
    return value; 
}

template <>
__device__ half convert_type<half>(float value) {
    return __float2half(value);
}

template <>
__device__ __nv_bfloat16 convert_type<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

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

template <typename dst_t>
static __global__ void dequantize_block_q5_K(const void *__restrict__ vx, dst_t *__restrict__ yy)
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
    y[0] = convert_type<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
    y[1] = convert_type<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = convert_type<dst_t>(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
    y[33] = convert_type<dst_t>(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
}

template <typename dst_t>
static __global__ void dequantize_block_q6_K(const void *__restrict__ vx, dst_t *__restrict__ yy)
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

    y[0] = convert_type<dst_t>(d * sc[0] * ((int8_t)((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = convert_type<dst_t>(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = convert_type<dst_t>(d * sc[4] * ((int8_t)((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = convert_type<dst_t>(d * sc[6] * ((int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

template <typename dst_t>
static void dequantize_row_q5_K_cuda(const void *vx, dst_t *y, const int k, cudaStream_t stream)
{
    const int nb = k / QK_K;
    dequantize_block_q5_K<dst_t><<<nb, 64, 0, stream>>>(vx, y);
}

template <typename dst_t>
static void dequantize_row_q6_K_cuda(const void *vx, dst_t *y, const int k, cudaStream_t stream)
{
    const int nb = k / QK_K;
    dequantize_block_q6_K<dst_t><<<nb, 64, 0, stream>>>(vx, y);
}

template <typename T>
using to_t_cuda_t = void (*)(const void *__restrict__ x, T *__restrict__ y, int k, cudaStream_t stream);
typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;
typedef to_t_cuda_t<__nv_bfloat16> to_bf16_cuda_t;
to_fp16_cuda_t ggml_get_to_fp16_cuda(int size_k, int quant_bytes_per_row)
{
    if (size_k % 256 != 0)
        throw std::runtime_error("size_k must be a multiple of 256");

    if (quant_bytes_per_row == size_k / 256 * 176)
        return dequantize_row_q5_K_cuda<half>;
    else if (quant_bytes_per_row == size_k / 256 * 210)
        return dequantize_row_q6_K_cuda<half>;
    else
        throw std::runtime_error("Unsupported weight_bytes_per_row");
}

static __device__ __forceinline__ float warp_reduce_sum(float x)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __global__ void quantize_q8_1(const float *__restrict__ x, void *__restrict__ vy, const int kx, const int kx_padded)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;

    if (ix >= kx_padded)
    {
        return;
    }

    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    const int i_padded = iy * kx_padded + ix;

    block_q8_1 *y = (block_q8_1 *)vy;

    const int ib = i_padded / QK8_1;  // block index
    const int iqs = i_padded % QK8_1; // quant index

    const float xi = ix < kx ? x[iy * kx + ix] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0)
    {
        return;
    }

    reinterpret_cast<half &>(y[ib].ds.x) = __float2half(d);
    reinterpret_cast<half &>(y[ib].ds.y) = __float2half(sum);
}

static void quantize_row_q8_1_cuda(const float *x, void *vy, const int kx, const int ky, const int kx_padded, cudaStream_t stream)
{
    const int block_num_x = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ky, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx, kx_padded);
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int *__restrict__ vl, const int *__restrict__ vh, const int *__restrict__ u, const uint8_t *__restrict__ sc,
    const uint8_t *__restrict__ m, const half2 &dm5, const float *__restrict__ d8)
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i)
    {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = __dp4a(v0i, u[2 * i + 0], __dp4a(v1i, u[2 * i + 1], 0));               // SIMD dot product
        const int dot2 = __dp4a(0x01010101, u[2 * i + 0], __dp4a(0x01010101, u[2 * i + 1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1, const int &iqs)
{
    const block_q5_K *bq5_K = (const block_q5_K *)vbq;

    int vl[2];
    int vh[2];
    int u[2 * QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int *ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int *qh = (const int *)(bq5_K->qh + 4 * ((iqs / 2) % 4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t *scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2)
    {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    }
    else
    {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t *sc = (const uint8_t *)aux;
    const uint8_t *m = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i)
    {
        const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t *x8, const int &i32)
{
    const uint16_t *x16 = (const uint16_t *)(x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] << 0;
    x32 |= x16[1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_from_int8_aligned(const int8_t *x8, const int &i32)
{
    return *((const int *)(x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int &vl, const int &vh, const int *__restrict__ u, const int8_t *__restrict__ scales,
    const float &d, const float *__restrict__ d8)
{
    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i)
    {
        const int sc = scales[4 * i];

        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;

        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d * sumf;
}


static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1, const int &iqs)
{

    const block_q6_K *bq6_K = (const block_q6_K *)vbq;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;

    const int8_t *scales = bq6_K->scales + scale_offset;

    int u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i)
    {
        u[i] = get_int_from_int8_aligned(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
        d8[i] = __half2float(__low2half(bq8_1[bq8_offset + 2 * i].ds));
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, __half2float(bq6_K->d), d8);
}

typedef float (*vec_dot_q_cuda_t)(const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1, const int &iqs);
template <int ncols_y, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__launch_bounds__((ncols_y <= 4 ? 4 : 2) * WARP_SIZE, 1)
static __global__ void mul_mat_vec_q(
    const void *__restrict__ vx, const void *__restrict__ vy, float *__restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst)
{
    constexpr int nwarps = ncols_y <= 4 ? 4 : 2;
    constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block * blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

    // partial sum for each thread
    float tmp[ncols_y][rows_per_cuda_block] = {0.0f};

    const block_q_t *x = (const block_q_t *)vx;
    const block_q8_1 *y = (const block_q8_1 *)vy;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter)
    {
        const int kby = kbx * (qk / QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi / vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j)
        {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i)
            {
                tmp[j][i] += vec_dot_q_cuda(
                    &x[kbx + (row0 + i) * blocks_per_row_x], &y[j * blocks_per_col_y + kby], kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0)
    {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j)
        {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i)
            {
                tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0)
    {
        return;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_y; ++j)
    {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i)
        {
#pragma unroll
            for (int l = 0; l < nwarps - 1; ++l)
            {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum(tmp[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block)
        {
            dst[j * nrows_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}


template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot>
static void mul_mat_vec_q_cuda(
    const void *vx, const void *vy, float *dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream)
{
    GGML_ASSERT(ncols_x % qk == 0);
    GGML_ASSERT(ncols_y <= MMVQ_MAX_BATCH_SIZE);

    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    int64_t nwarps = 1;
    int64_t rows_per_cuda_block = 1;

    // if (g_device_caps[id].cc < CC_RDNA2)
    if (1)
    { // NVIDIA and AMD older than RDNA2
        switch (ncols_y)
        {
        case 1:
            nwarps = 4;
            rows_per_cuda_block = 1;
            break;
        case 2:
        case 3:
        case 4:
            nwarps = 4;
            rows_per_cuda_block = 2;
            break;
        case 5:
        case 6:
        case 7:
        case 8:
            nwarps = 2;
            rows_per_cuda_block = 2;
            break;
        default:
            GGML_ASSERT(false);
            break;
        }
    }
    const int64_t nblocks = (nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const dim3 block_nums(nblocks, 1, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    switch (ncols_y)
    {
    case 1:
        mul_mat_vec_q<1, qk, qi, block_q_t, vdr, vec_dot>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
        break;
    case 2:
        mul_mat_vec_q<2, qk, qi, block_q_t, vdr, vec_dot>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
        break;
    case 3:
        mul_mat_vec_q<3, qk, qi, block_q_t, vdr, vec_dot>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
        break;
    case 4:
        mul_mat_vec_q<4, qk, qi, block_q_t, vdr, vec_dot>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
        break;
    case 5:
        mul_mat_vec_q<5, qk, qi, block_q_t, vdr, vec_dot>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
        break;
    case 6:
        mul_mat_vec_q<6, qk, qi, block_q_t, vdr, vec_dot>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
        break;
    case 7:
        mul_mat_vec_q<7, qk, qi, block_q_t, vdr, vec_dot>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
        break;
    case 8:
        mul_mat_vec_q<8, qk, qi, block_q_t, vdr, vec_dot>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
        break;
    default:
        GGML_ASSERT(false);
        break;
    }
}


static void ggml_cuda_op_mul_mat_vec_q(
    int src0_type, // 13: GGML_TYPE_Q5_K, 14: GGML_TYPE_Q6_K
    const int64_t m, const int64_t n, const int64_t k, const int64_t k_padded,
    const char *src0_dd_i, const char *src1_ddq_i, float *dst_dd_i, 
    cudaStream_t stream)
{
    // m: ne01 = weight->ne[1], n: ne11 = input->ne[1], k: ne00 = ne10 = input->ne[0] = weight->ne[0]
    if (!(k % QK8_1 == 0))
        throw std::runtime_error("ne10 must be a multiple of QK8_1");

    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    switch (src0_type)
    {
    case 13: // GGML_TYPE_Q5_K
        mul_mat_vec_q_cuda<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(src0_dd_i, src1_ddq_i, dst_dd_i, k, m, k_padded, n, m, stream);
        break;
    case 14: // GGML_TYPE_Q6_K:
        mul_mat_vec_q_cuda<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(src0_dd_i, src1_ddq_i, dst_dd_i, k, m, k_padded, n, m, stream);
        break;
    default:
        GGML_ASSERT(false);
        break;
    }
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

    const int m = b_weight.size(0);
    const int n = a.size(0);
    const int k = a.size(1);

    const auto scalar_type = a.scalar_type();
    if (scalar_type != at::kFloat&&  scalar_type != at::kHalf && scalar_type != at::kBFloat16)
        throw std::runtime_error("Unsupported scalar type");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    if (n > MMVQ_MAX_BATCH_SIZE)
    {
        auto options = torch::TensorOptions().dtype(at::kHalf).device(a.device());
        at::Tensor c = torch::empty({n, m}, options);
        at::Tensor temp_dq = torch::empty({m, k}, options);
        auto a_half = scalar_type == at::kHalf ? a : a.to(at::kHalf);

        vllm::gguf::reconstruct_gemm
        (
            at::cuda::getCurrentCUDABlasHandle(),
            (const half*) a_half.data_ptr(),
            (const uint8_t*) b_weight.data_ptr(),
            (half*) c.data_ptr(),
            (half*) temp_dq.data_ptr(),
            m,  // m
            n,  // n
            k,  // k
            b_weight.size(1) // quant_bytes_per_row
        );

        return scalar_type == at::kHalf ? c : c.to(scalar_type);
    }
    else {
        const int64_t k_padded = GGML_PAD(a.size(1), MATRIX_ROW_PADDING);

        // quantize input tensor(a) 
        auto opt_aq = torch::TensorOptions().dtype(at::kByte).device(a.device());
        at::Tensor aq = torch::empty({a.size(0), k_padded * 36 / 32}, opt_aq);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        auto a_float = scalar_type == at::kFloat ? a : a.to(at::kFloat);
        vllm::gguf::quantize_row_q8_1_cuda((const float*)a_float.data_ptr(), aq.data_ptr(), k, n, k_padded, stream);

        // src0_type 
        int src0_type = 0;
        auto quant_bytes_per_row = b_weight.size(1);
        if (quant_bytes_per_row == k / 256 * 176)
            src0_type = 13;
        else if (quant_bytes_per_row == k / 256 * 210)
            src0_type = 14;
        else
            throw std::runtime_error("Unsupported weight_bytes_per_row");

        auto opt_c = torch::TensorOptions().dtype(at::kFloat).device(a.device());
        at::Tensor c = torch::empty({n, m}, opt_c);

        vllm::gguf::ggml_cuda_op_mul_mat_vec_q(src0_type, m, n, k, k_padded, (const char*)b_weight.data_ptr(), (const char*)aq.data_ptr(), (float *)c.data_ptr(), stream);
        return scalar_type == at::kFloat ? c : c.to(scalar_type);
    }
}
