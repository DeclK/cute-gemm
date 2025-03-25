#include <cublas_v2.h>
#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>

#include <cute/tensor.hpp>

#include "detail/data.h"
#include <cute/util/gpu_clock.hpp>

template <typename Config>
__global__ void /* __launch_bounds__(128, 1) */
gemm_multi_stage(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
                 int k) {
  using namespace cute;
  using X = Underscore;

  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutC = typename Config::SmemLayoutC;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;
  using S2GCopyAtomC = typename Config::S2GCopyAtomC;
  using S2GCopyC = typename Config::S2GCopyC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix = blockIdx.x;
  int iy = blockIdx.y;

  // use Tensor notation to represent device pointer + dimension
  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));  // (M, K)
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));  // (N, K)
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));  // (M, N)

  // slice the tensor to small one which is used for current thread block.
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));  // (kTileM, kTileK, k), (128, 32, 1)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));  // (kTileN, kTileK, k), (128, 32, 1)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix)); // (kTileM, kTileN), (128, 128)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{});  // (kTileM, kTileK, kStage) (128, 32, 5)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{});  // (kTileN, kTileK, kStage) (128, 32, 5).

  // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
  // method
  TiledMMA tiled_mma; // deal with (32, 16, 16) sized mma
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K), (8, 4, 2) = ((2,2,2), 128/32, 32/16)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K), (4, 8, 2) = ((2,2), 128/16, 32/16)
  auto tCrD = thr_mma.partition_fragment_C(gD);           // (MMA, MMA_M, MMA_N), (4, 4, 8) = ((2,2), 128/32, 128/16)
  
#if 0
  if (thread0() && block0()) {
    print("tCrA shape = "); print(tCrA); print("\n");
    print("tCrB shape = "); print(tCrB); print("\n");
    print("tCrD shape = "); print(tCrD); print("\n");
  }
#endif

  // fill zero for accumulator
  clear(tCrD);

  // gmem -cp.async-> shm, sA.shape = (128, 32, 5), gA.shape = (128, 32, k)
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k), (8, 4, 1, 1) = ((8,1), 128/32, 32/32, k)
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage), (8, 4, 1, 3) = ((8,1), 128/32, 32/32, kStage)
  
  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k), (8, 4, 1, 1) = ((8,1), 128/32, 32/32, k)
  auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage) (8, 4, 1, 3) = ((8,1), 128/32, 32/32, kStage)

#if 0
  if (thread0() && block0()) {
    print("tAgA_copy shape = "); print(tAgA_copy); print("\n");
    print("tAsA_copy shape = "); print(tAsA_copy); print("\n");
    print("tBgB_copy shape = "); print(tBgB_copy); print("\n");
    print("tBsB_copy shape = "); print(tBsB_copy); print("\n");
  }
#endif
  
  // shm -ldmatrix-> reg, sA.shape = (128, 32, 5)
  // this seems similar to use tiled_mma to partition the tensor
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);  // (CPY, CPY_M, CPY_K, kStage), (8, 4, 2, 3) = ((8,1), 128/32, 32/16, kStage)
                                               // 8 * 128 != 32 * 16, this means, there are elements repeated in the copy process
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, CPY_M, CPY_K), (8, 4, 2) = ((8,1), 128/32, 32/16)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);  // ? (CPY, CPY_M, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);  // ? (CPY, CPY_M, CPY_K)

#if 0
  if (thread0() && block0()) {
    print("tAsA shape = "); print(tAsA); print("\n");
    print("tCrA_view shape = "); print(tCrA_view); print("\n");
    print("tBsB shape = "); print(tBsB); print("\n");
    print("tCrB_view shape = "); print(tCrB_view); print("\n");
  }
#endif


  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;

  // submit kStage - 1 tile, gmem -> shm
  // what happens if istage is bigger than COPY_K?
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
               tAsA_copy(_, _, _, istage));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
               tBsB_copy(_, _, _, istage));
    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();
  __syncthreads();

  int ik = 0;
  // smem -> reg
  cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // I assume the pipelined loop will be represented as a high level API in cutlass, so that we can focus on the algorithm itself
  // loop over k: i. load tile, ii. mma
  int ntile = k / kTileK;
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    // this nk is what we called small k iter = kTileK // MMA_Tile_K
    // where as big k iter = ntile
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      // To make it easier to understand, inside this loop,
      // the ik usually means the small k iter
      int ik_next = (ik + 1) % nk;

      if (ik == 0) {
        // if this is the first small k iter, load the next tile of the big k iter
        // why we don't load this when we submit the first kStage - 1 tile?
        if (itile_to_read < ntile) {
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read), tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, ismem_write));
          cp_async_fence();
          
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }
      }

      if (ik == nk - 1) {
        // if this is the last small k iter, make sure the next big k iter is ready to be loaded.
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      // shm -> reg s[itile][ik + 1] -> r[ik + 1], this is asynchronous
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

      // how do we know the tCrA and tCrB are ready to be used?
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }  // for ik
  }    // itile

  // use less shared memory as a scratchpad tile to use large wide instuction
  // Dreg -> shm -> reg -> global
  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});    // (128, 128, 2)

  // still use tiled_mma's C layout to partition the tensor, just like s2r_tiled_copy_a
  // in a warp perspective, one tiled copy would deal with (32, 32) sized matrix
  // and it will iterate to copy the whole (128, 128) sized matrix
  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);   // (CPY, CPY_M, CPY_N), (8, 4, 4) = ((2, (2, 2)), 128/32, 128/32)
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe), (8, 1, 1, 2) = ((2, (2, 2)), 1, 1, pipe)

  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe), (8, 1, 2, 2) = ((8, 1), 1, 1, pipe)
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);  // (CPY, CPY_M, CPY_N), (8, 4, 4) = ((8, 1), 128/32, 128/32)

  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN), (8, 16) = ((8, 1), 128/32 * 128/32), why CPY_, it is the same as CPY
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN), (8, 16) = ((2, (2, 2)), 128/32 * 128/32)

#if 0
  if (thread0() && block0()) {
    print("tCrC_r2s shape = "); print(tCrC_r2s); print("\n");
    print("tCsC_r2s shape = "); print(tCsC_r2s); print("\n");
    print("tCsC_s2g shape = "); print(tCsC_s2g); print("\n");
    print("tCgC_s2g shape = "); print(tCgC_s2g); print("\n");
    print("tCgC_s2gx shape = "); print(tCgC_s2gx); print("\n");
    print("tCrC_r2sx shape = "); print(tCrC_r2sx); print("\n");
  }
#endif

  int step = size<3>(tCsC_r2s);  // pipe = 2
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {  // for (int i = 0; i < 16; i += 2), how does this make a difference with
                                                        // for (int i = 0; i < 16; i += 1)
    // reg -> shm
#pragma unroll
    for (int j = 0; j < step; ++j) {
      // we add a temp tensor to cope with accumulator and output data type difference
      auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
      cute::copy(tCrC_r2sx(_, i + j), t);

      cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();

#pragma unroll
    // shm -> global
    for (int j = 0; j < step; ++j) {
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }

    __syncthreads();
  }
}

namespace config {

using namespace cute;

template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
          int kStage_ = 5, int kSmemLayoutCBatch_ = 2,
          typename ComputeType = T_>
struct GemmConfig {
  using T = T_;

  // tile configuration
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;
  static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;

  using SmemLayoutAtom = decltype(composition(Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
                                              make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                                                          make_stride(Int<kTileK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
                                             make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{}, 
                                             make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;

  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  // EU means execution unit
  // After repeating the MMA will deal with 32x16x16 sized mma
  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

  // Define the TiledCopy from global memory to shared memory
  // Each Tile will copy 32x32 half_t elements
  using G2SCopyA = decltype(make_tiled_copy(g2s_copy_atom{},
                                            make_layout(make_shape(Int<32>{}, Int<4>{}),
                                                        make_stride(Int<4>{}, Int<1>{})),
                                            make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  // shared memory to register copy
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;

  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  // epilogue: register to global via shared memory
  using SmemLayoutAtomC = decltype(composition(Swizzle<2, 3, 3>{},
                                               make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}), 
                                                           make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomC{},
                                             make_shape(Int<kMmaPM>{}, 
                                                        Int<kMmaPN>{}, 
                                                        Int<kSmemLayoutCBatch>{})));

  static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{}),
                "C shared memory request is large than A's one pipe");

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;

  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC = decltype(make_tiled_copy(S2GCopyAtomC{},
                                            make_layout(make_shape(Int<32>{}, Int<4>{}),
                                                        make_stride(Int<4>{}, Int<1>{})),
                                            make_layout(make_shape(Int<1>{}, Int<8>{}))));

  static constexpr int kThreadNum = size(MMA{});
  static constexpr int shm_size_AB = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});

  static constexpr int kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(T);
};

}  // namespace config

int main(int argc, char *argv[]) {
  using T = cute::half_t;
  using namespace cute;
  using X = Underscore;

  srand(10086);

  cublasHandle_t handle;
  cublasCreate(&handle);
  int cublas_version;
  cublasGetVersion_v2(handle, &cublas_version);
  printf("cuBLAS version: %d\n", cublas_version);

  // default;
  int M = 128;
  int N = 128;
  int K = 32;

  int num_iter = 1;

  using ComputeType = T;

  T *Aptr;
  T *Bptr;
  T *Dptr;
  T *Dptr_cublas;

  T *Aptr_host;
  T *Bptr_host;
  T *Dptr_host;
  T *Dptr_host_blas;

  // allocate memory
  Aptr_host = (T *)malloc(sizeof(T) * M * K);
  Bptr_host = (T *)malloc(sizeof(T) * N * K);
  Dptr_host = (T *)malloc(sizeof(T) * M * N);

  Dptr_host_blas = (T *)malloc(sizeof(T) * M * N);

  cudaMalloc(&Aptr, sizeof(T) * M * K);
  cudaMalloc(&Bptr, sizeof(T) * N * K);
  cudaMalloc(&Dptr, sizeof(T) * M * N);
  cudaMalloc(&Dptr_cublas, sizeof(T) * M * N);

  // create random tensor on host
  auto tA = make_tensor(Aptr_host, make_shape(M, K), make_stride(K, 1));
  auto tB = make_tensor(Bptr_host, make_shape(N, K), make_stride(K, 1));
  auto tD = make_tensor(Dptr_host, make_shape(M, N), make_stride(N, 1));
  cpu_rand_data(&tA);
  cpu_rand_data(&tB);
  clear(tD);

  // copy tensor to device
  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * N * K, cudaMemcpyHostToDevice);
  cudaMemcpy(Dptr, Dptr_host, sizeof(T) * M * N, cudaMemcpyHostToDevice);
  cudaMemset(Dptr_cublas, 0, sizeof(T) * M * N);

  // gemm config
  config::GemmConfig<T, 128, 128, 32, 3> gemm_config;
  // config::GemmConfig<T, 128, 128, 32, 3>::S2GCopyC s2g_tiled_copy_c;
  // print(s2g_tiled_copy_c);
  // config::GemmConfig<T, 128, 128, 32, 3>::G2SCopyA g2s_tiled_copy_a;
  // print(g2s_tiled_copy_a);
  // print_latex(g2s_tiled_copy_a);
  // print(typename decltype(gemm_config)::MMA{});

  // kernel config
  dim3 block = gemm_config.kThreadNum;
  dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
            (M + gemm_config.kTileM - 1) / gemm_config.kTileM);
  int shm_size = gemm_config.kShmSize;
  printf("block = (%d, %d), gird = (%d, %d), shm = %d\n", block.x, block.y, grid.x, grid.y, shm_size);

  half alpha = 1.f;
  half beta = 0.f;
  GPU_Clock timer;

  // cute multi-stage gemm
  timer.start();
  for (int it = 0; it < num_iter; ++it) {
    // cudaMemset(Dptr, 0, sizeof(T) * M * N);
    cudaFuncSetAttribute(gemm_multi_stage<decltype(gemm_config)>, 
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    gemm_multi_stage<decltype(gemm_config)>
    <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
  }
  cudaDeviceSynchronize();
  double elapsed_time_ms = timer.milliseconds() / num_iter;
  printf("elapsed time cute = %f ms\n", elapsed_time_ms);

  // blas
  timer.start();
  for (int it = 0; it < num_iter; ++it) {
    // cudaMemset(Dptr_cublas, 0, sizeof(T) * M * N);
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                                     &alpha, (half *)Bptr, K, (half *)Aptr, K,
                                     &beta, (half *)Dptr_cublas, N);
  }
  cudaDeviceSynchronize();
  elapsed_time_ms = timer.milliseconds() / num_iter;
  printf("elapsed time cublas = %f ms\n", elapsed_time_ms);

  // copy result back to host
  cudaMemcpy(Dptr_host, Dptr, sizeof(T) * M * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(Dptr_host_blas, Dptr_cublas, sizeof(T) * M * N, cudaMemcpyDeviceToHost);

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf_fail("err = %d, str = %s\n", err, cudaGetErrorString(err));
  }

  // check blas result with cute result
  gpu_compare(Dptr, Dptr_cublas, M * N);

  // show part of the output tensor
  auto tD_host = make_tensor(Dptr_host, make_shape(M, N), make_stride(N, 1));
  auto tD_host_blas = make_tensor(Dptr_host_blas, make_shape(M, N), make_stride(N, 1));

  auto tile = make_tile(min(8, M), min(8, N));
  auto t32x32 = local_tile(tD_host, tile, make_coord(0, 0));
  auto t32x32_blas = local_tile(tD_host_blas, tile, make_coord(0, 0));

  printf("M = %d, N = %d, K = %d\n", M, N, K);

  // printf("our-impl:\n");
  // print_tensor(t32x32);
  // printf("cublas:\n");
  // print_tensor(t32x32_blas);

}
