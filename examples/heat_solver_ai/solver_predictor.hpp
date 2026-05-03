// solver_predictor.hpp
// =====================================================================
// 用 LibTorch 加载训练好的 solver_selector.pt（TorchScript），
// 在 C++ 里基于热传导问题特征预测最优求解器。
//
// 公开接口刻意只依赖 std + MFEM 的 SparseMatrix；只有 .cpp 才 #include
// <torch/script.h>，便于使用方按需链接 libtorch。
//
// 列表（问题/网格/求解器）的顺序必须与 train_solver_selector.py 中保持
// 严格一致；同样数值特征的拼接顺序也写死在 .cpp 的 BuildFeatureVector()
// 中——任何调整都需双侧同步修改。
// =====================================================================

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mfem { class SparseMatrix; }

namespace mfem_ai {

// 描述单个热传导问题——既是预测器输入，也是 benchmark 的特征记录。
struct HeatProblemDescriptor {
   // 类别（必须出自 SolverPredictor::Problems()/MeshTypes() 列表）
   std::string problem;          // "iso" / "aniso" / "var" / "react" / "trans" / "conv"
   std::string mesh_type;        // "quad" / "tri" / "hex" / "tet"

   // 几何/离散
   int    dim         = 2;
   int    poly_order  = 1;
   int    n_dof       = 0;
   int    n_elements  = 0;

   // 矩阵特征（可由 ExtractMatrixFeatures 自动填充）
   long   nnz             = 0;
   double nnz_per_row     = 0.0;
   double sparsity        = 0.0;
   double symmetry_ratio  = 0.0;
   double diag_dominance  = 0.0;
   bool   is_spd          = false;
   double frob_norm_est   = 0.0;

   // 物理参数
   double kappa_iso     = 1.0;
   double aniso_ratio   = 1.0;
   double reaction_coef = 0.0;
   double dt_step       = 0.0;
   double peclet        = 0.0;
};

// 从已组装的 SparseMatrix 抽取与 heat_bench 一致的统计量，
// 写回 desc.nnz / sparsity / symmetry_ratio 等字段。
void ExtractMatrixFeatures(const mfem::SparseMatrix& A,
                           HeatProblemDescriptor& desc);

class SolverPredictor {
public:
   // 加载 TorchScript 模型；构造失败抛 std::runtime_error。
   explicit SolverPredictor(const std::string& torchscript_path);
   ~SolverPredictor();
   SolverPredictor(const SolverPredictor&)            = delete;
   SolverPredictor& operator=(const SolverPredictor&) = delete;

   // 单次预测，返回求解器名称（如 "PCG_GS"）。
   std::string Predict(const HeatProblemDescriptor& d) const;

   // 返回每个求解器的 softmax 得分，按得分降序。
   std::vector<std::pair<std::string, double>>
   PredictAll(const HeatProblemDescriptor& d) const;

   // 类别表（与 Python 训练脚本完全一致）
   static const std::vector<std::string>& Problems();
   static const std::vector<std::string>& MeshTypes();
   static const std::vector<std::string>& Solvers();

   // 数值特征顺序（与 Python NUMERIC_FEATURES 一致，调试用）
   static const std::vector<std::string>& NumericFeatureNames();

private:
   struct Impl;
   std::unique_ptr<Impl> impl_;
};

} // namespace mfem_ai
