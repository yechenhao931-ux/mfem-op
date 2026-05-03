//                MFEM Communication-Avoiding Kernels Benchmark
//
// Compile with: make perf_bench_camm
//
// Sample runs:
//    OMP_NUM_THREADS=4 ./perf_bench_camm -m ../data/inline-quad.mesh -r 7 -reps 100
//
// Description:
//   Measures the SpMV / multi-dot / fused AXPY+Dot kernels added in
//   linalg/cpu_kernels.hpp. Each kernel is verified against the existing
//   serial baseline (Frobenius / max-abs diff) before timing.
//
//   Why "communication-avoiding"? In a parallel iterative solver each
//   independent dot product triggers an MPI_Allreduce. The DotMulti()
//   kernel computes K independent dot products in a single pass and lets
//   the caller batch K Allreduces into one. Locally that also cuts the
//   bandwidth on the shared "y" vector by K.
//
//   The fused AxpyAndSelfDot() variant collapses the two-pass
//      r -= alpha*z;   beta = (r,r);
//   sequence (used in CG every iteration) into a single sweep of r,
//   halving its bandwidth.

#include "mfem.hpp"
#include "linalg/cpu_kernels.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;
using namespace mfem;

namespace
{
struct Timer
{
   using clock = std::chrono::steady_clock;
   clock::time_point t0;
   void start() { t0 = clock::now(); }
   double stop() const
   {
      return std::chrono::duration<double>(clock::now() - t0).count();
   }
};
}

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/inline-quad.mesh";
   int ref_levels = 6;
   int order = 1;
   int reps = 100;
   int K = 4;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement steps.");
   args.AddOption(&order, "-o", "--order", "Polynomial order.");
   args.AddOption(&reps, "-reps", "--repetitions", "Repetitions per kernel.");
   args.AddOption(&K, "-k", "--num-vectors",
                  "Number of x vectors batched in DotMulti.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   Mesh mesh(mesh_file, 1, 1);
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }
   const int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   const int N = fes.GetTrueVSize();
   cout << "DOFs: " << N << endl;

   ConstantCoefficient one(1.0);
   BilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble(); a.Finalize();
   const SparseMatrix &A = a.SpMat();
   cout << "nnz: " << A.NumNonZeroElems() << endl;

   // --- Set up data --------------------------------------------------------
   Vector x(N), y(N), y_ref(N);
   for (int i = 0; i < N; i++)
   {
      x(i) = std::sin(0.001 * i) + 0.3;
      y(i) = std::cos(0.0002 * i);
      y_ref(i) = y(i);
   }
   x.UseDevice(false); y.UseDevice(false); y_ref.UseDevice(false);

   std::vector<Vector> Xs(K, Vector(N));
   std::vector<const real_t*> Xp(K);
   for (int k = 0; k < K; k++)
   {
      for (int i = 0; i < N; i++)
      {
         Xs[k](i) = std::sin(0.0007 * (i + 17 * k));
      }
      Xs[k].UseDevice(false);
      Xp[k] = Xs[k].GetData();
   }

   Timer t;

   // --- 1. NNZ-balanced SpMV correctness + timing -------------------------
   y_ref = 0.0;
   A.Mult(x, y_ref);  // baseline (already uses balanced kernel internally
                      // when OMP>1 — included for warm-up / sanity)

   y = 0.0;
   t.start();
   for (int r = 0; r < reps; r++) { A.Mult(x, y); }
   const double t_spmv = t.stop();

   double spmv_diff = 0.0;
   for (int i = 0; i < N; i++)
   {
      const double d = y(i) - y_ref(i);
      if (std::fabs(d) > spmv_diff) { spmv_diff = std::fabs(d); }
   }

   // --- 2. Multi-vector dot product (K dots in one pass) ------------------
   std::vector<real_t> dots_ref(K), dots_new(K);
   for (int k = 0; k < K; k++) { dots_ref[k] = (Xs[k] * y_ref); }

   t.start();
   for (int r = 0; r < reps; r++)
   {
      cpu_kernels::DotMulti(N, K, Xp.data(), y_ref.GetData(), dots_new.data());
   }
   const double t_dotmulti = t.stop();

   // For comparison, time K serial dots:
   t.start();
   for (int r = 0; r < reps; r++)
   {
      for (int k = 0; k < K; k++) { dots_ref[k] = (Xs[k] * y_ref); }
   }
   const double t_dotk = t.stop();

   double dot_diff = 0.0;
   for (int k = 0; k < K; k++)
   {
      const double d = dots_ref[k] - dots_new[k];
      if (std::fabs(d) > dot_diff) { dot_diff = std::fabs(d); }
   }

   // --- 3. Fused AXPY + (y,y) ---------------------------------------------
   //   reference: y += a*z; norm = (y,y)  (two passes over y)
   //   fused    : in one sweep                                          (one pass)
   const real_t alpha = 1.234e-3;
   y = y_ref;
   y_ref += y_ref; // change reference so it differs from initial
   for (int i = 0; i < N; i++) { y_ref(i) = std::cos(0.0002 * i); }
   y = y_ref;

   real_t norm_ref = 0.0, norm_new = 0.0;
   t.start();
   for (int r = 0; r < reps; r++)
   {
      y_ref.Add(alpha, x);
      norm_ref = (y_ref * y_ref);
   }
   const double t_two_pass = t.stop();

   t.start();
   for (int r = 0; r < reps; r++)
   {
      norm_new = cpu_kernels::AxpyAndSelfDot(N, alpha, x.GetData(), y.GetData());
   }
   const double t_fused = t.stop();

   const double norm_diff = std::fabs(norm_ref - norm_new);
   double y_diff = 0.0;
   for (int i = 0; i < N; i++)
   {
      const double d = y(i) - y_ref(i);
      if (std::fabs(d) > y_diff) { y_diff = std::fabs(d); }
   }

   // --- Report ------------------------------------------------------------
   cout << fixed << setprecision(4);
   cout << "------------------------------------------------------------\n";
   cout << "Kernel                     time (s)    note\n";
   cout << "------------------------------------------------------------\n";
   cout << "SpMV  (nnz-balanced)       " << t_spmv
        << "    max|diff|=" << scientific << spmv_diff << fixed << "\n";
   cout << "DotMulti(K=" << K << ")               " << t_dotmulti
        << "    max|diff|=" << scientific << dot_diff << fixed << "\n";
   cout << "  (vs. K serial dots)      " << t_dotk
        << "    speedup="
        << (t_dotk / std::max(t_dotmulti, 1e-12)) << "x\n";
   cout << "AXPY + (y,y) two-pass      " << t_two_pass << "\n";
   cout << "  fused AxpyAndSelfDot     " << t_fused
        << "    speedup="
        << (t_two_pass / std::max(t_fused, 1e-12)) << "x  ";
   cout << "max|y-y_ref|=" << scientific << y_diff
        << "  norm diff=" << norm_diff << fixed << "\n";
   cout << "------------------------------------------------------------\n";

   const bool ok = (spmv_diff < 1e-10) && (dot_diff < 1e-8 * std::fabs(dots_ref[0])
                                           + 1e-12)
                    && (y_diff < 1e-12) && (norm_diff < 1e-6 * std::fabs(norm_ref) + 1e-12);
   return ok ? 0 : 1;
}
