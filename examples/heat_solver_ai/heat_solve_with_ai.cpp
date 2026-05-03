// heat_solve_with_ai.cpp
// =====================================================================
// 演示：在 MFEM 中组装一个稳态热传导问题，把矩阵+物理参数喂给训练好的
// SolverSelector，按推荐求解器求解并比较前 3 名的实测耗时。
//
// 用法（需先用 train_solver_selector.py 生成 solver_selector.pt）：
//   ./heat_solve_with_ai
//   ./heat_solve_with_ai --model solver_selector.pt --problem aniso \
//                        --mesh-type quad --refine 4 --order 1
// =====================================================================

#include "mfem.hpp"
#include "solver_predictor.hpp"

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;
using namespace mfem_ai;

using Clock = std::chrono::high_resolution_clock;
inline double ms_since(Clock::time_point t) {
   return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// 与 heat_bench.cpp 同口径的各向异性张量
class AnisoTensor : public MatrixCoefficient {
   double kx_, ky_, kz_;
public:
   AnisoTensor(int dim, double kx, double ky, double kz = 1.0)
      : MatrixCoefficient(dim), kx_(kx), ky_(ky), kz_(kz) {}
   void Eval(DenseMatrix& K, ElementTransformation&,
             const IntegrationPoint&) override {
      const int d = GetWidth(); K.SetSize(d); K = 0.0;
      K(0, 0) = kx_;
      if (d > 1) K(1, 1) = ky_;
      if (d > 2) K(2, 2) = kz_;
   }
};

double VarKappa(const Vector& x) {
   double r2 = (x(0) - 0.5) * (x(0) - 0.5);
   if (x.Size() > 1) r2 += (x(1) - 0.5) * (x(1) - 0.5);
   if (x.Size() > 2) r2 += (x(2) - 0.5) * (x(2) - 0.5);
   return 1.0 + 9.0 * std::exp(-r2 / 0.05);
}

// ----------------------------------------------------------------------
// 按求解器名称运行一次 —— 与 heat_bench 中 RunSolver 同语义
// ----------------------------------------------------------------------
struct SolveResult {
   bool   converged = false;
   int    iters = 0;
   double time_ms = 0.0;
   double rel_res = 0.0;
};

SolveResult Solve(const std::string& name, SparseMatrix& A,
                  const Vector& B, Vector& x,
                  double rtol, double atol, int max_iter, int kdim)
{
   SolveResult r;
   std::unique_ptr<Solver> prec;
   if (name == "PCG_Jacobi" || name == "GMRES_Jacobi" ||
       name == "BiCGSTAB_Jacobi") prec.reset(new DSmoother(A, 0));
   else if (name == "PCG_GS")     prec.reset(new GSSmoother(A));
   else if (name == "PCG_Cheby")  prec.reset(new DSmoother(A, 2));

   auto t0 = Clock::now();
   if (name == "CG" || name == "PCG_Jacobi" || name == "PCG_GS" ||
       name == "PCG_Cheby") {
      CGSolver cg; cg.SetOperator(A);
      cg.SetRelTol(rtol); cg.SetAbsTol(atol);
      cg.SetMaxIter(max_iter); cg.SetPrintLevel(0);
      if (prec) cg.SetPreconditioner(*prec);
      cg.Mult(B, x);
      r.converged = cg.GetConverged(); r.iters = cg.GetNumIterations();
      r.rel_res   = cg.GetFinalNorm();
   }
   else if (name == "MINRES") {
      MINRESSolver mr; mr.SetOperator(A);
      mr.SetRelTol(rtol); mr.SetAbsTol(atol);
      mr.SetMaxIter(max_iter); mr.SetPrintLevel(0);
      mr.Mult(B, x);
      r.converged = mr.GetConverged(); r.iters = mr.GetNumIterations();
      r.rel_res   = mr.GetFinalNorm();
   }
   else if (name == "GMRES" || name == "GMRES_Jacobi") {
      GMRESSolver gm; gm.SetOperator(A);
      gm.SetRelTol(rtol); gm.SetAbsTol(atol);
      gm.SetMaxIter(max_iter); gm.SetKDim(kdim); gm.SetPrintLevel(0);
      if (prec) gm.SetPreconditioner(*prec);
      gm.Mult(B, x);
      r.converged = gm.GetConverged(); r.iters = gm.GetNumIterations();
      r.rel_res   = gm.GetFinalNorm();
   }
   else if (name == "BiCGSTAB" || name == "BiCGSTAB_Jacobi") {
      BiCGSTABSolver bs; bs.SetOperator(A);
      bs.SetRelTol(rtol); bs.SetAbsTol(atol);
      bs.SetMaxIter(max_iter); bs.SetPrintLevel(0);
      if (prec) bs.SetPreconditioner(*prec);
      bs.Mult(B, x);
      r.converged = bs.GetConverged(); r.iters = bs.GetNumIterations();
      r.rel_res   = bs.GetFinalNorm();
   }
#ifdef MFEM_USE_SUITESPARSE
   else if (name == "DIRECT_UMF") {
      UMFPackSolver direct; direct.SetOperator(A);
      direct.Mult(B, x);
      r.converged = true; r.iters = 1; r.rel_res = 0.0;
   }
#endif
   r.time_ms = ms_since(t0);
   return r;
}

int main(int argc, char** argv)
{
   // 命令行参数
   std::string model_path = "solver_selector.pt";
   std::string problem    = "iso";    // iso / aniso / var / react / conv
   std::string mesh_type  = "quad";   // quad / tri / hex / tet
   int    refine    = 4;
   int    order     = 1;
   double rtol      = 1e-8, atol = 1e-12;
   int    max_iter  = 3000, kdim = 50;
   bool   verify    = true;          // 是否实测前 3 名

   for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&]() { return std::string(argv[++i]); };
      if      (a == "--model"     && i+1<argc) model_path = next();
      else if (a == "--problem"   && i+1<argc) problem    = next();
      else if (a == "--mesh-type" && i+1<argc) mesh_type  = next();
      else if (a == "--refine"    && i+1<argc) refine     = std::stoi(next());
      else if (a == "--order"     && i+1<argc) order      = std::stoi(next());
      else if (a == "--rtol"      && i+1<argc) rtol       = std::stod(next());
      else if (a == "--no-verify")             verify     = false;
   }

   // 1) 构造网格
   std::cout << "构造 " << mesh_type << " 网格，refine=" << refine << "\n";
   std::unique_ptr<Mesh> mesh;
   if (mesh_type == "quad") {
      mesh.reset(new Mesh(Mesh::MakeCartesian2D(
         2, 2, Element::QUADRILATERAL, true)));
   } else if (mesh_type == "tri") {
      mesh.reset(new Mesh(Mesh::MakeCartesian2D(
         2, 2, Element::TRIANGLE, true)));
   } else if (mesh_type == "hex") {
      mesh.reset(new Mesh(Mesh::MakeCartesian3D(
         2, 2, 2, Element::HEXAHEDRON)));
   } else if (mesh_type == "tet") {
      mesh.reset(new Mesh(Mesh::MakeCartesian3D(
         2, 2, 2, Element::TETRAHEDRON)));
   } else {
      std::cerr << "未知 mesh-type: " << mesh_type << "\n";
      return 1;
   }
   for (int i = 0; i < refine; ++i) mesh->UniformRefinement();

   const int dim = mesh->Dimension();
   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(mesh.get(), &fec);

   // 2) 组装该热传导问题
   std::vector<std::unique_ptr<Coefficient>>       scalar_keep;
   std::vector<std::unique_ptr<VectorCoefficient>> vec_keep;
   std::vector<std::unique_ptr<MatrixCoefficient>> mat_keep;

   BilinearForm a(&fes);
   LinearForm   b(&fes);
   auto* one = new ConstantCoefficient(1.0);
   scalar_keep.emplace_back(one);
   b.AddDomainIntegrator(new DomainLFIntegrator(*one));

   HeatProblemDescriptor d;
   d.problem    = problem;
   d.mesh_type  = mesh_type;
   d.dim        = dim;
   d.poly_order = order;

   if (problem == "iso") {
      d.kappa_iso = 1.0;
      auto* k = new ConstantCoefficient(d.kappa_iso);
      scalar_keep.emplace_back(k);
      a.AddDomainIntegrator(new DiffusionIntegrator(*k));
   } else if (problem == "aniso") {
      double kx = 100.0, ky = 1.0, kz = 1.0;
      d.kappa_iso   = (kx + ky + kz) / 3.0;
      d.aniso_ratio = std::max({kx, ky, kz}) / std::min({kx, ky, kz});
      auto* K = new AnisoTensor(dim, kx, ky, kz);
      mat_keep.emplace_back(K);
      a.AddDomainIntegrator(new DiffusionIntegrator(*K));
   } else if (problem == "var") {
      d.kappa_iso = 5.5;
      auto* k = new FunctionCoefficient(VarKappa);
      scalar_keep.emplace_back(k);
      a.AddDomainIntegrator(new DiffusionIntegrator(*k));
   } else if (problem == "react") {
      d.kappa_iso     = 1.0;
      d.reaction_coef = 10.0;
      auto* k = new ConstantCoefficient(d.kappa_iso);
      auto* r = new ConstantCoefficient(d.reaction_coef);
      scalar_keep.emplace_back(k);
      scalar_keep.emplace_back(r);
      a.AddDomainIntegrator(new DiffusionIntegrator(*k));
      a.AddDomainIntegrator(new MassIntegrator(*r));
   } else if (problem == "trans") {
      d.kappa_iso = 1.0;
      d.dt_step   = 1e-3;
      auto* dtcf = new ConstantCoefficient(d.dt_step);
      scalar_keep.emplace_back(dtcf);
      a.AddDomainIntegrator(new MassIntegrator());
      a.AddDomainIntegrator(new DiffusionIntegrator(*dtcf));
   } else if (problem == "conv") {
      d.kappa_iso = 0.01;
      Vector beta(dim); beta = 0.0; beta(0) = 1.0;
      if (dim > 1) beta(1) = 0.5;
      d.peclet = beta.Norml2() * mesh->GetElementSize(0) / (2.0 * d.kappa_iso);
      auto* k  = new ConstantCoefficient(d.kappa_iso);
      auto* bc = new VectorConstantCoefficient(beta);
      scalar_keep.emplace_back(k);
      vec_keep.emplace_back(bc);
      a.AddDomainIntegrator(new DiffusionIntegrator(*k));
      a.AddDomainIntegrator(new ConvectionIntegrator(*bc, -1.0));
   } else {
      std::cerr << "未知 problem: " << problem << "\n";
      return 1;
   }

   a.Assemble(); a.Finalize();
   b.Assemble();

   GridFunction x_sol(&fes); x_sol = 0.0;
   Array<int> ess_tdof;
   if (mesh->bdr_attributes.Size()) {
      Array<int> ess_bdr(mesh->bdr_attributes.Max()); ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof);
   }
   OperatorPtr A; Vector X, B;
   a.FormLinearSystem(ess_tdof, x_sol, b, A, X, B);
   SparseMatrix& A_sp = static_cast<SparseMatrix&>(*A.Ptr());

   // 3) 填充矩阵特征
   d.n_dof      = fes.GetTrueVSize();
   d.n_elements = mesh->GetNE();
   ExtractMatrixFeatures(A_sp, d);

   std::cout << "  DOF=" << d.n_dof
             << "  nnz=" << d.nnz
             << "  sym_ratio=" << std::fixed << std::setprecision(3)
             << d.symmetry_ratio
             << "  is_spd=" << (d.is_spd ? "yes" : "no") << "\n";

   // 4) 调用预测器
   std::cout << "\n加载模型: " << model_path << "\n";
   SolverPredictor pred(model_path);
   auto ranking = pred.PredictAll(d);

   std::cout << "\n推荐排序 (softmax 概率):\n";
   for (size_t i = 0; i < std::min<size_t>(ranking.size(), 5); ++i) {
      std::cout << "  " << (i+1) << ". "
                << std::left << std::setw(18) << ranking[i].first
                << "  " << std::fixed << std::setprecision(3)
                << ranking[i].second << "\n";
   }
   std::string chosen = ranking.front().first;
   std::cout << "\n→ 选择: " << chosen << "\n";

   // 5) 用推荐求解器实际求解
   {
      Vector x(X.Size()); x = 0.0;
      auto t0 = Clock::now();
      auto r = Solve(chosen, A_sp, B, x, rtol, atol, max_iter, kdim);
      double total = ms_since(t0);
      std::cout << std::fixed << std::setprecision(2)
                << "[" << chosen << "] iters=" << r.iters
                << "  time=" << total << " ms"
                << "  converged=" << (r.converged ? "yes" : "no") << "\n";
   }

   // 6) 可选验证：实测前 3 名，看预测是否真的最快
   if (verify && ranking.size() >= 3) {
      std::cout << "\n=== 验证：前 3 推荐实测耗时 ===\n";
      for (int i = 0; i < 3; ++i) {
         const std::string& name = ranking[i].first;
#ifndef MFEM_USE_SUITESPARSE
         if (name == "DIRECT_UMF") continue;
#endif
         Vector x(X.Size()); x = 0.0;
         auto t0 = Clock::now();
         auto r = Solve(name, A_sp, B, x, rtol, atol, max_iter, kdim);
         double total = ms_since(t0);
         std::cout << "  " << std::left << std::setw(18) << name
                   << "  iters=" << std::setw(5) << r.iters
                   << "  time=" << std::fixed << std::setprecision(2)
                   << total << " ms"
                   << "  " << (r.converged ? "✓" : "✗") << "\n";
      }
   }
   return 0;
}
