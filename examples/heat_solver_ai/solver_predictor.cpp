// solver_predictor.cpp — LibTorch + MFEM 推理实现

#include "solver_predictor.hpp"

#include "mfem.hpp"
#include <torch/script.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfem_ai {

// ----------------------------------------------------------------------
// 静态类别表（顺序必须与 train_solver_selector.py 一致）
// ----------------------------------------------------------------------
namespace {
const std::vector<std::string> kProblems = {
   "iso", "aniso", "var", "react", "trans", "conv"
};
const std::vector<std::string> kMeshTypes = {
   "quad", "tri", "hex", "tet"
};
const std::vector<std::string> kSolvers = {
   "CG", "PCG_Jacobi", "PCG_GS", "PCG_Cheby",
   "MINRES", "GMRES", "GMRES_Jacobi",
   "BiCGSTAB", "BiCGSTAB_Jacobi", "DIRECT_UMF"
};
// 与 Python 端 NUMERIC_FEATURES 一一对应
const std::vector<std::string> kNumericNames = {
   "dim", "poly_order",
   "log_n_dof", "log_n_elements", "log_nnz",
   "nnz_per_row", "sparsity", "symmetry_ratio", "diag_dominance",
   "is_spd", "log_frob_norm",
   "log_kappa", "log_aniso", "reaction_coef",
   "log_dt", "log_peclet",
};

inline double safe_log10(double x, double eps = 1e-12) {
   return std::log10(std::max(x, eps));
}

int IndexOrThrow(const std::vector<std::string>& vocab,
                 const std::string& key, const char* what) {
   auto it = std::find(vocab.begin(), vocab.end(), key);
   if (it == vocab.end()) {
      throw std::invalid_argument(std::string("未知 ") + what + ": " + key);
   }
   return static_cast<int>(it - vocab.begin());
}
} // namespace

const std::vector<std::string>& SolverPredictor::Problems()   { return kProblems; }
const std::vector<std::string>& SolverPredictor::MeshTypes()  { return kMeshTypes; }
const std::vector<std::string>& SolverPredictor::Solvers()    { return kSolvers; }
const std::vector<std::string>& SolverPredictor::NumericFeatureNames()
                                                              { return kNumericNames; }

// ----------------------------------------------------------------------
// 矩阵特征提取（与 heat_bench.cpp::ComputeMatStats 计算口径相同）
// ----------------------------------------------------------------------
void ExtractMatrixFeatures(const mfem::SparseMatrix& A,
                           HeatProblemDescriptor& d)
{
   const int n = A.Height();
   const int* I = A.GetI();
   const int* J = A.GetJ();
   const double* V = A.GetData();
   const long nnz = A.NumNonZeroElems();

   d.nnz         = nnz;
   d.nnz_per_row = (double) nnz / std::max(n, 1);
   d.sparsity    = 1.0 - (double) nnz / (double(n) * double(n));

   double asym = 0.0, norm = 0.0, frob_sq = 0.0;
   int taken = 0, max_take = 4000;
   for (int row = 0; row < n; ++row) {
      for (int k = I[row]; k < I[row+1]; ++k) {
         double v = V[k];
         frob_sq += v * v;
         if (taken < max_take && J[k] != row) {
            double aji = A.Elem(J[k], row);
            asym += std::abs(v - aji);
            norm += std::abs(v) + std::abs(aji) + 1e-30;
            ++taken;
         }
      }
   }
   d.symmetry_ratio = (norm > 1e-30) ? asym / norm : 0.0;
   d.frob_norm_est  = std::sqrt(frob_sq);

   double sum_dd = 0.0;
   int spd_rows = 0;
   for (int row = 0; row < n; ++row) {
      double diag = 0.0, off = 0.0;
      for (int k = I[row]; k < I[row+1]; ++k) {
         if (J[k] == row) diag = V[k];
         else             off += std::abs(V[k]);
      }
      if (diag > off) ++spd_rows;
      sum_dd += (off > 1e-30) ? diag / off : 10.0;
   }
   d.diag_dominance = sum_dd / std::max(n, 1);
   d.is_spd         = (spd_rows == n) && (d.symmetry_ratio < 1e-2);
}

// ----------------------------------------------------------------------
// 特征向量拼装：[数值特征] + [problem one-hot] + [mesh one-hot]
// 必须与 Python build_training_set 的顺序逐位一致。
// ----------------------------------------------------------------------
static std::vector<float> BuildFeatureVector(const HeatProblemDescriptor& d)
{
   const int n_numeric = (int) kNumericNames.size();
   std::vector<float> f;
   f.reserve(n_numeric + kProblems.size() + kMeshTypes.size());

   f.push_back((float) d.dim);
   f.push_back((float) d.poly_order);
   f.push_back((float) safe_log10((double) std::max(d.n_dof, 1)));
   f.push_back((float) safe_log10((double) std::max(d.n_elements, 1)));
   f.push_back((float) safe_log10((double) std::max<long>(d.nnz, 1)));
   f.push_back((float) d.nnz_per_row);
   f.push_back((float) d.sparsity);
   f.push_back((float) d.symmetry_ratio);
   f.push_back((float) std::min(d.diag_dominance, 100.0));
   f.push_back(d.is_spd ? 1.0f : 0.0f);
   f.push_back((float) safe_log10(std::max(d.frob_norm_est, 1e-30)));
   f.push_back((float) safe_log10(d.kappa_iso));
   f.push_back((float) safe_log10(d.aniso_ratio));
   f.push_back((float) d.reaction_coef);
   f.push_back((float) safe_log10(d.dt_step + 1e-12));
   f.push_back((float) safe_log10(d.peclet  + 1e-12));

   const int p_idx = IndexOrThrow(kProblems,  d.problem,   "problem");
   const int m_idx = IndexOrThrow(kMeshTypes, d.mesh_type, "mesh_type");
   for (size_t i = 0; i < kProblems.size();  ++i)
      f.push_back(i == (size_t) p_idx ? 1.0f : 0.0f);
   for (size_t i = 0; i < kMeshTypes.size(); ++i)
      f.push_back(i == (size_t) m_idx ? 1.0f : 0.0f);
   return f;
}

// ----------------------------------------------------------------------
// PIMPL：藏 torch 头
// ----------------------------------------------------------------------
struct SolverPredictor::Impl {
   torch::jit::script::Module module;
   int feature_dim = 0;
};

SolverPredictor::SolverPredictor(const std::string& path)
   : impl_(new Impl())
{
   try {
      impl_->module = torch::jit::load(path);
      impl_->module.eval();
   } catch (const c10::Error& e) {
      throw std::runtime_error("加载 TorchScript 模型失败: " + path
                               + "\n  " + e.what());
   }
   impl_->feature_dim = (int)(kNumericNames.size()
                              + kProblems.size()
                              + kMeshTypes.size());
}

SolverPredictor::~SolverPredictor() = default;

std::vector<std::pair<std::string, double>>
SolverPredictor::PredictAll(const HeatProblemDescriptor& d) const
{
   auto feat = BuildFeatureVector(d);
   if ((int) feat.size() != impl_->feature_dim) {
      throw std::runtime_error("特征维度不匹配: 实际="
                               + std::to_string(feat.size())
                               + " 期望="
                               + std::to_string(impl_->feature_dim));
   }

   torch::NoGradGuard ng;
   torch::Tensor input = torch::from_blob(
      feat.data(), {1, (int64_t) feat.size()}, torch::kFloat32).clone();

   std::vector<torch::jit::IValue> inputs;
   inputs.emplace_back(input);
   torch::Tensor logits = impl_->module.forward(inputs).toTensor();
   torch::Tensor probs  = torch::softmax(logits, /*dim=*/1).contiguous();

   const float* p = probs.data_ptr<float>();
   std::vector<std::pair<std::string, double>> out;
   out.reserve(kSolvers.size());
   for (size_t i = 0; i < kSolvers.size(); ++i) {
      out.emplace_back(kSolvers[i], (double) p[i]);
   }
   std::sort(out.begin(), out.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });
   return out;
}

std::string SolverPredictor::Predict(const HeatProblemDescriptor& d) const
{
   return PredictAll(d).front().first;
}

} // namespace mfem_ai
