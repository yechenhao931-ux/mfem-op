// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CPU_KERNELS_HPP
#define MFEM_CPU_KERNELS_HPP

// -----------------------------------------------------------------------------
// Host-only kernels optimized for serial / shared-memory CPU execution.
//
// The MFEM device-portable kernels (in linalg/kernels.hpp and the
// mfem::forall machinery) target both CPU and GPU and therefore avoid
// host-specific pragmas. The kernels in this header are dedicated CPU paths
// that combine three optimization layers:
//
//   1. Thread-level parallelism via OpenMP `parallel for`
//      (only emitted when the library is built with MFEM_USE_OPENMP).
//   2. Data-level parallelism via `omp simd` reductions and aligned
//      contiguous-stride loops, which let modern auto-vectorizers emit
//      AVX2 / AVX-512 / NEON / SVE FMA instructions.
//   3. Cache-friendly access patterns: row-static partitioning for CSR
//      SpMV and column-major streaming for dense GEMV.
//
// They are entry points used by SparseMatrix and DenseMatrix when the
// data lives in host memory and the active backend is the CPU. The
// device path is left untouched.
// -----------------------------------------------------------------------------

#include "../config/config.hpp"
#include <cmath>
#include <cstddef>

#ifdef MFEM_USE_OPENMP
#include <omp.h>
#endif

// Portable "restrict" qualifier for the host kernels. Avoid colliding with
// MFEM_CPU_RESTRICT used elsewhere — these are internal to this header.
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define MFEM_CPU_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define MFEM_CPU_RESTRICT __restrict
#else
#define MFEM_CPU_RESTRICT
#endif

namespace mfem
{
namespace cpu_kernels
{

// Threshold below which it is not worth spinning up OpenMP threads. Using
// threads on tiny problems is dominated by fork/join overhead.
inline constexpr int omp_threshold = 512;

// ---------------------------------------------------------------------------
// CSR SpMV: y += a * A * x
// ---------------------------------------------------------------------------
inline void CSRAddMult(const int height,
                       const int *MFEM_CPU_RESTRICT Ip,
                       const int *MFEM_CPU_RESTRICT Jp,
                       const real_t *MFEM_CPU_RESTRICT Ap,
                       const real_t *MFEM_CPU_RESTRICT xp,
                       real_t *MFEM_CPU_RESTRICT yp,
                       const real_t a)
{
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel for schedule(static) if (height >= omp_threshold)
#endif
   for (int i = 0; i < height; i++)
   {
      const int rb = Ip[i];
      const int re = Ip[i+1];
      real_t d = 0.0;
      // Inner loop has an indirect gather A[j]*x[J[j]]; modern compilers
      // emit vgather (AVX-512) or scalarize cleanly. The simd hint also
      // unrolls and exposes FMA pipelines.
      #if defined(MFEM_USE_OPENMP)
      #pragma omp simd reduction(+:d)
      #endif
      for (int j = rb; j < re; j++)
      {
         d += Ap[j] * xp[Jp[j]];
      }
      yp[i] += a * d;
   }
}

// ---------------------------------------------------------------------------
// CSR transpose SpMV: y += a * A^T * x
// Race-free per-thread accumulation buffers, reduced at the end.
// ---------------------------------------------------------------------------
inline void CSRAddMultTranspose(const int height,
                                const int width,
                                const int *MFEM_CPU_RESTRICT Ip,
                                const int *MFEM_CPU_RESTRICT Jp,
                                const real_t *MFEM_CPU_RESTRICT Ap,
                                const real_t *MFEM_CPU_RESTRICT xp,
                                real_t *MFEM_CPU_RESTRICT yp,
                                const real_t a)
{
#if defined(MFEM_USE_OPENMP)
   if (height >= omp_threshold)
   {
      const int nthreads = omp_get_max_threads();
      // Per-thread scratch in a single contiguous buffer. We avoid std::vector
      // here so the dependency stays on stack-friendly allocation.
      real_t *scratch = new real_t[(std::size_t)nthreads * width]();
      #pragma omp parallel
      {
         const int tid = omp_get_thread_num();
         real_t *yloc = scratch + (std::size_t)tid * width;
         #pragma omp for schedule(static) nowait
         for (int i = 0; i < height; i++)
         {
            const real_t xi = a * xp[i];
            const int rb = Ip[i];
            const int re = Ip[i+1];
            for (int j = rb; j < re; j++)
            {
               yloc[Jp[j]] += Ap[j] * xi;
            }
         }
         // Reduce all thread-local buffers into yp. Parallelize across the
         // output dimension; each thread owns a disjoint column range.
         #pragma omp barrier
         #pragma omp for schedule(static)
         for (int k = 0; k < width; k++)
         {
            real_t s = 0.0;
            for (int t = 0; t < nthreads; t++)
            {
               s += scratch[(std::size_t)t * width + k];
            }
            yp[k] += s;
         }
      }
      delete [] scratch;
      return;
   }
#endif
   // Serial fallback. The inner loop has a write dependency on yp[Jp[j]],
   // so we cannot use omp simd here, but auto-vectorization for the
   // gather-scatter is still worthwhile on hot caches.
   for (int i = 0; i < height; i++)
   {
      const real_t xi = a * xp[i];
      const int rb = Ip[i];
      const int re = Ip[i+1];
      for (int j = rb; j < re; j++)
      {
         yp[Jp[j]] += Ap[j] * xi;
      }
   }
}

// ---------------------------------------------------------------------------
// CSR |A| * x: y += sum_j |A_ij| * x_j
// ---------------------------------------------------------------------------
inline void CSRAddAbsMult(const int height,
                          const int *MFEM_CPU_RESTRICT Ip,
                          const int *MFEM_CPU_RESTRICT Jp,
                          const real_t *MFEM_CPU_RESTRICT Ap,
                          const real_t *MFEM_CPU_RESTRICT xp,
                          real_t *MFEM_CPU_RESTRICT yp)
{
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel for schedule(static) if (height >= omp_threshold)
#endif
   for (int i = 0; i < height; i++)
   {
      const int rb = Ip[i];
      const int re = Ip[i+1];
      real_t d = 0.0;
      #if defined(MFEM_USE_OPENMP)
      #pragma omp simd reduction(+:d)
      #endif
      for (int j = rb; j < re; j++)
      {
         d += std::fabs(Ap[j]) * xp[Jp[j]];
      }
      yp[i] += d;
   }
}

// ---------------------------------------------------------------------------
// Vector AXPY: y[i] += a * x[i]
// ---------------------------------------------------------------------------
inline void Axpy(const int N, const real_t a,
                 const real_t *MFEM_CPU_RESTRICT x,
                 real_t *MFEM_CPU_RESTRICT y)
{
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel for simd schedule(static) if (N >= omp_threshold)
#endif
   for (int i = 0; i < N; i++)
   {
      y[i] += a * x[i];
   }
}

// ---------------------------------------------------------------------------
// Dot product: returns sum_i x[i]*y[i]
// ---------------------------------------------------------------------------
inline real_t Dot(const int N,
                  const real_t *MFEM_CPU_RESTRICT x,
                  const real_t *MFEM_CPU_RESTRICT y)
{
   real_t s = 0.0;
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel for simd reduction(+:s) schedule(static) \
      if (N >= omp_threshold)
#endif
   for (int i = 0; i < N; i++)
   {
      s += x[i] * y[i];
   }
   return s;
}

// ---------------------------------------------------------------------------
// Vector sum: returns sum_i x[i]
// ---------------------------------------------------------------------------
inline real_t Sum(const int N, const real_t *MFEM_CPU_RESTRICT x)
{
   real_t s = 0.0;
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel for simd reduction(+:s) schedule(static) \
      if (N >= omp_threshold)
#endif
   for (int i = 0; i < N; i++)
   {
      s += x[i];
   }
   return s;
}

// ---------------------------------------------------------------------------
// Vector min / max: assume N > 0.
// ---------------------------------------------------------------------------
inline real_t Min(const int N, const real_t *MFEM_CPU_RESTRICT x)
{
   real_t m = x[0];
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel for simd reduction(min:m) schedule(static) \
      if (N >= omp_threshold)
#endif
   for (int i = 0; i < N; i++)
   {
      m = (x[i] < m) ? x[i] : m;
   }
   return m;
}

inline real_t Max(const int N, const real_t *MFEM_CPU_RESTRICT x)
{
   real_t m = x[0];
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel for simd reduction(max:m) schedule(static) \
      if (N >= omp_threshold)
#endif
   for (int i = 0; i < N; i++)
   {
      m = (x[i] > m) ? x[i] : m;
   }
   return m;
}

// ---------------------------------------------------------------------------
// Dense GEMV (column-major): y += a * A * x, where A is height x width with
// data stored column-major (MFEM's DenseMatrix layout). The outer column loop
// is sequential, the inner row loop is contiguous-stride: ideal for SIMD.
// Threading splits the row range so writes to y[] are disjoint.
// ---------------------------------------------------------------------------
inline void DenseAddMult(const int height, const int width,
                         const real_t *MFEM_CPU_RESTRICT data,
                         const real_t *MFEM_CPU_RESTRICT x,
                         real_t *MFEM_CPU_RESTRICT y,
                         const real_t a = 1.0)
{
#if defined(MFEM_USE_OPENMP)
   if (height >= omp_threshold)
   {
      #pragma omp parallel
      {
         const int nthreads = omp_get_num_threads();
         const int tid = omp_get_thread_num();
         const int chunk = (height + nthreads - 1) / nthreads;
         const int rlo = tid * chunk;
         const int rhi = (rlo + chunk < height) ? rlo + chunk : height;
         for (int col = 0; col < width; col++)
         {
            const real_t xc = a * x[col];
            const real_t *MFEM_CPU_RESTRICT d_col = data + (std::size_t)col * height;
            #pragma omp simd
            for (int row = rlo; row < rhi; row++)
            {
               y[row] += xc * d_col[row];
            }
         }
      }
      return;
   }
#endif
   for (int col = 0; col < width; col++)
   {
      const real_t xc = a * x[col];
      const real_t *MFEM_CPU_RESTRICT d_col = data + (std::size_t)col * height;
      #if defined(MFEM_USE_OPENMP)
      #pragma omp simd
      #endif
      for (int row = 0; row < height; row++)
      {
         y[row] += xc * d_col[row];
      }
   }
}

// ---------------------------------------------------------------------------
// NNZ-balanced row partitioning for parallel CSR SpMV.
//
// Static row partitioning (height/nthreads rows per thread) hands every
// thread the same number of rows but a wildly different amount of work
// when the matrix has a non-uniform sparsity pattern (refined or AMR
// meshes, high-order elements). Splitting at row boundaries that
// equalise the per-thread *nnz count* gives each thread roughly the
// same amount of FMA work and removes load imbalance, which translates
// to nearly linear strong scaling for memory-resident matrices.
//
// Output:
//   row_split[0..nthreads] is filled such that
//     row_split[0] == 0, row_split[nthreads] == height
//   and Ip[row_split[t+1]] - Ip[row_split[t]] is approximately
//   nnz/nthreads for each thread t.
// The split is computed by binary search over the prefix-sum I[].
// O(nthreads * log(height)) — negligible vs. one SpMV.
// ---------------------------------------------------------------------------
inline void CSRPartitionByNNZ(const int height,
                              const int *MFEM_CPU_RESTRICT Ip,
                              const int nthreads,
                              int *MFEM_CPU_RESTRICT row_split)
{
   row_split[0] = 0;
   row_split[nthreads] = height;
   if (nthreads <= 1) { return; }
   const long long nnz = Ip[height];
   for (int t = 1; t < nthreads; t++)
   {
      const long long target = (nnz * t) / nthreads;
      // Binary search for smallest r in [0, height] with Ip[r] >= target.
      int lo = 0, hi = height;
      while (lo < hi)
      {
         const int mid = (lo + hi) >> 1;
         if ((long long)Ip[mid] < target) { lo = mid + 1; }
         else                              { hi = mid; }
      }
      // Keep splits monotone (defensive — should already hold).
      if (lo < row_split[t-1]) { lo = row_split[t-1]; }
      row_split[t] = lo;
   }
}

// CSR SpMV with a precomputed nnz-balanced row partition.
// row_split has length nthreads+1 (use CSRPartitionByNNZ to fill it).
// For matrices with a uniform row length the result is identical to
// CSRAddMult; for irregular matrices the parallel speedup approaches
// nthreads instead of being capped by the slowest row band.
inline void CSRAddMultBalanced(const int /*height*/,
                               const int *MFEM_CPU_RESTRICT Ip,
                               const int *MFEM_CPU_RESTRICT Jp,
                               const real_t *MFEM_CPU_RESTRICT Ap,
                               const real_t *MFEM_CPU_RESTRICT xp,
                               real_t *MFEM_CPU_RESTRICT yp,
                               const real_t a,
                               const int nthreads,
                               const int *MFEM_CPU_RESTRICT row_split)
{
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel num_threads(nthreads)
   {
      const int tid = omp_get_thread_num();
      const int rb = row_split[tid];
      const int re = row_split[tid + 1];
      for (int i = rb; i < re; i++)
      {
         const int rs = Ip[i];
         const int rsp = Ip[i+1];
         real_t d = 0.0;
         #pragma omp simd reduction(+:d)
         for (int j = rs; j < rsp; j++)
         {
            d += Ap[j] * xp[Jp[j]];
         }
         yp[i] += a * d;
      }
   }
#else
   (void)nthreads; (void)row_split;
   const int H = row_split ? row_split[nthreads] : 0;
   for (int i = 0; i < H; i++)
   {
      const int rs = Ip[i], re = Ip[i+1];
      real_t d = 0.0;
      for (int j = rs; j < re; j++) { d += Ap[j] * xp[Jp[j]]; }
      yp[i] += a * d;
   }
#endif
}

// ---------------------------------------------------------------------------
// Multi-vector dot product (single-pass, single-Allreduce friendly).
//
//   out[k] = sum_i x[k][i] * y[i]   for k = 0 .. K-1
//
// In a parallel iterative solver each global dot product triggers an
// MPI_Allreduce; collapsing K logically-independent dots into a single
// kernel lets the caller do *one* batched Allreduce on a length-K
// buffer instead of K. On the compute side it also amortises the read
// of y[] across K dots, halving (or quartering) the bandwidth bill.
//
// Memory layout: x is an array of K vector pointers, each of length N.
// ---------------------------------------------------------------------------
inline void DotMulti(const int N, const int K,
                     const real_t *const *MFEM_CPU_RESTRICT x,
                     const real_t *MFEM_CPU_RESTRICT y,
                     real_t *MFEM_CPU_RESTRICT out)
{
   for (int k = 0; k < K; k++) { out[k] = 0.0; }
#if defined(MFEM_USE_OPENMP)
   if (N >= omp_threshold)
   {
      const int nt = omp_get_max_threads();
      // Per-thread accumulators. Keep the buffer cache-line padded to
      // avoid false sharing between threads on the same K-row.
      const int pad = 8; // 64 B / 8 B per real_t
      const int stride = ((K + pad - 1) / pad) * pad;
      real_t *acc = new real_t[(std::size_t)nt * stride]();
      #pragma omp parallel
      {
         const int tid = omp_get_thread_num();
         real_t *a_local = acc + (std::size_t)tid * stride;
         #pragma omp for schedule(static) nowait
         for (int i = 0; i < N; i++)
         {
            const real_t yi = y[i];
            for (int k = 0; k < K; k++)
            {
               a_local[k] += x[k][i] * yi;
            }
         }
      }
      for (int t = 0; t < nt; t++)
      {
         for (int k = 0; k < K; k++)
         {
            out[k] += acc[(std::size_t)t * stride + k];
         }
      }
      delete [] acc;
      return;
   }
#endif
   for (int i = 0; i < N; i++)
   {
      const real_t yi = y[i];
      for (int k = 0; k < K; k++)
      {
         out[k] += x[k][i] * yi;
      }
   }
}

// ---------------------------------------------------------------------------
// Fused AXPY + dot:  y += a * x;  return (y, y) computed *after* the AXPY.
//
// In CG inner loops we typically run
//     r -= alpha * z;          // single sweep over r and z
//     betanom = (r, r);        // second sweep over r
// fusing both into one streaming kernel halves the bandwidth on r[].
// ---------------------------------------------------------------------------
inline real_t AxpyAndSelfDot(const int N, const real_t a,
                             const real_t *MFEM_CPU_RESTRICT x,
                             real_t *MFEM_CPU_RESTRICT y)
{
   real_t s = 0.0;
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel for simd reduction(+:s) schedule(static) \
      if (N >= omp_threshold)
#endif
   for (int i = 0; i < N; i++)
   {
      const real_t yi = y[i] + a * x[i];
      y[i] = yi;
      s += yi * yi;
   }
   return s;
}

// ---------------------------------------------------------------------------
// Diagonal-scaled SpMV (Jacobi preconditioner application + apply A in one
// pass). Computes y += a * D^{-1} A x, where D is the inverse diagonal
// already supplied as Dinv[].
//
// Useful inside Krylov solvers with a Jacobi preconditioner: the
// classical implementation needs three streams (z = A x, then z *= Dinv,
// then y += alpha z). This kernel folds them into a single sweep over
// rows, halving traffic on z and removing one allocation.
// ---------------------------------------------------------------------------
inline void CSRDinvAddMult(const int height,
                           const int *MFEM_CPU_RESTRICT Ip,
                           const int *MFEM_CPU_RESTRICT Jp,
                           const real_t *MFEM_CPU_RESTRICT Ap,
                           const real_t *MFEM_CPU_RESTRICT Dinv,
                           const real_t *MFEM_CPU_RESTRICT xp,
                           real_t *MFEM_CPU_RESTRICT yp,
                           const real_t a)
{
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel for schedule(static) if (height >= omp_threshold)
#endif
   for (int i = 0; i < height; i++)
   {
      const int rb = Ip[i];
      const int re = Ip[i+1];
      real_t d = 0.0;
      #if defined(MFEM_USE_OPENMP)
      #pragma omp simd reduction(+:d)
      #endif
      for (int j = rb; j < re; j++)
      {
         d += Ap[j] * xp[Jp[j]];
      }
      yp[i] += a * Dinv[i] * d;
   }
}

} // namespace cpu_kernels
} // namespace mfem

#endif // MFEM_CPU_KERNELS_HPP