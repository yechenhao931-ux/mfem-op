//                  MFEM Threaded BilinearForm::Assemble Benchmark
//
// Compile with: make perf_bench_assemble
//
// Sample runs:
//    OMP_NUM_THREADS=4 ./perf_bench_assemble -m ../data/inline-quad.mesh -r 7
//    OMP_NUM_THREADS=8 ./perf_bench_assemble -m ../data/inline-hex.mesh  -r 4 -o 2
//
// Description: Measures BilinearForm::Assemble timing with and without
//              EnableThreadedAssembly() and verifies that the resulting
//              global stiffness matrix is identical (Frobenius norm of
//              the difference == 0).

#include "mfem.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>

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

double FrobDiff(const SparseMatrix &A, const SparseMatrix &B)
{
   MFEM_VERIFY(A.Height() == B.Height() && A.Width() == B.Width(), "shape");
   const int *Ai = A.HostReadI();
   const int *Bi = B.HostReadI();
   const int *Aj = A.HostReadJ();
   const int *Bj = B.HostReadJ();
   const real_t *Ad = A.HostReadData();
   const real_t *Bd = B.HostReadData();
   const int n = A.Height();
   real_t s = 0.0;
   for (int i = 0; i < n; i++)
   {
      MFEM_VERIFY(Ai[i+1]-Ai[i] == Bi[i+1]-Bi[i], "row size mismatch");
      for (int j = Ai[i]; j < Ai[i+1]; j++)
      {
         MFEM_VERIFY(Aj[j] == Bj[j], "col mismatch");
         const real_t d = Ad[j] - Bd[j];
         s += d * d;
      }
   }
   return std::sqrt(s);
}
}

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/inline-quad.mesh";
   int ref_levels = 6;
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement steps.");
   args.AddOption(&order, "-o", "--order", "Polynomial order.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   Mesh mesh(mesh_file, 1, 1);
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }
   const int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   cout << "DOFs: " << fes.GetTrueVSize()
        << "  elements: " << mesh.GetNE() << endl;

   ConstantCoefficient one(1.0);
   Timer t;

   // --- Serial baseline --------------------------------------------------
   BilinearForm a_serial(&fes);
   a_serial.UsePrecomputedSparsity(1);
   a_serial.AddDomainIntegrator(new DiffusionIntegrator(one));
   t.start();
   a_serial.Assemble();
   const double t_serial = t.stop();
   a_serial.Finalize();

   // --- Threaded ---------------------------------------------------------
   BilinearForm a_thread(&fes);
   a_thread.EnableThreadedAssembly(true);
   a_thread.AddDomainIntegrator(new DiffusionIntegrator(one));
   t.start();
   a_thread.Assemble();
   const double t_thread = t.stop();
   a_thread.Finalize();

   const double diff = FrobDiff(a_serial.SpMat(), a_thread.SpMat());

   cout << fixed << setprecision(4);
   cout << "------------------------------------------------------------\n";
   cout << "Assemble timings\n";
   cout << "  serial   : " << t_serial << " s\n";
   cout << "  threaded : " << t_thread << " s   ("
        << (t_serial / std::max(t_thread, 1e-12)) << "x speedup)\n";
   cout << "  ||A_serial - A_thread||_F = " << scientific
        << diff << fixed << "\n";
   cout << "------------------------------------------------------------\n";

   return diff < 1e-10 ? 0 : 1;
}
