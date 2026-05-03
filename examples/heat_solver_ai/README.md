# heat_solver_ai —— 基于 PyTorch 的 MFEM 热传导求解器自动选择

完整流水线包括三个阶段：

| 阶段 | 工具 | 产物 |
| ---- | ---- | ---- |
| 1. 数据采集 | `examples/heat_bench` (C++) | `heat_solver_data.csv` |
| 2. 模型训练 | `train_solver_selector.py` (Python) | `solver_selector.pt`、`solver_selector_meta.json` |
| 3. 推理使用 | `solver_predictor` + `heat_solve_with_ai` (C++) | 在线推荐求解器 |

---

## 1. 数据采集 (C++)

`examples/heat_bench.cpp` 覆盖 6 类热传导问题：

| 标签 | 物理 | 控制方程 |
| ---- | ---- | -------- |
| `iso`   | 稳态各向同性 | −∇·(κ∇u)=f |
| `aniso` | 稳态各向异性 | −∇·(K∇u)=f, K=diag(kₓ,kᵧ,k_z) |
| `var`   | 空间变系数  | −∇·(κ(x)∇u)=f |
| `react` | 反应–扩散   | −∇·(κ∇u)+r·u=f |
| `trans` | 瞬态隐式一步 | (M+dt·K)u = f |
| `conv`  | 对流–扩散   | −∇·(κ∇u)+β·∇u=f |

测试维度：

- 网格类型：`quad / tri`（2D），`hex / tet`（3D）
- 精炼级别：2D 0–5，3D 0–3
- 多项式阶：1–2
- 求解器：CG、PCG_Jacobi/GS/Cheby、MINRES、GMRES、GMRES_Jacobi、BiCGSTAB、BiCGSTAB_Jacobi、DIRECT_UMF（开了 SuiteSparse）

每行 CSV 记录：物理参数、矩阵特征（DOF、nnz、对称性、对角占优、SPD 启发判定…）、求解器、组装/求解耗时、迭代数、收敛状态。

构建并运行：

```bash
# 1) 先把 MFEM 串行版构建好（参见仓库根目录 INSTALL）
cd <mfem-root>/examples
make heat_bench                        # 已注册到 SEQ_EXAMPLES，make all 也能带上

# 2) 几秒钟联调：仅跑 iso × quad 几个尺度，确认管道通
./heat_bench --smoke --output smoke.csv

# 3) 完整跑一遍，几十秒到几分钟（受限于 max-ref）
./heat_bench --output heat_solver_data.csv

# 常用选项
./heat_bench --max-ref 4 --max-ref-3d 2 --no-direct
./heat_bench --problem iso          # 仅各向同性
./heat_bench --dim 2                # 仅 2D
```

产物：`heat_solver_data.csv`（每个 (问题, 网格, 求解器) 一行）。

---

## 2. 模型训练 (Python)

把 CSV 喂给 `train_solver_selector.py`。脚本会：

1. 按 `(problem, mesh_type, dim, poly_order, ref_level)` 分组，取耗时最短且收敛的求解器作为标签；
2. 数值特征（DOF/nnz/对称性…）取 log10 后做 z-score 标准化，标准化系数随模型一起保存（写入 `register_buffer`）；
3. 类别特征（problem、mesh_type）做 one-hot 拼接到数值特征后；
4. 训练一个两层 GELU MLP（隐层默认 96），Dropout 0.1，AdamW + 类别加权交叉熵；
5. 取最佳验证准确率的 checkpoint，trace 成 TorchScript 保存。

依赖（建议 conda/venv）：

```bash
pip install torch numpy pandas
```

运行：

```bash
cd examples/heat_solver_ai
python3 train_solver_selector.py --csv ../heat_solver_data.csv
# 可调参数
#   --epochs 300 --batch-size 128 --lr 5e-4 --hidden 128
```

产物：

- `solver_selector.pt` — TorchScript 模型（**特征标准化已嵌入**，C++ 端直接喂原始特征向量即可）
- `solver_selector_meta.json` — 类别表、特征顺序、Top-1 / Top-3 验证准确率
- `training_report.txt` — 验证集 Top-1/Top-3 + 混淆矩阵

### 2.1 评估脚本

光看准确率不够，因为「预测错」不一定意味着「慢得要命」——选了次优求解器
但耗时只多了 5% 是可以接受的。`evaluate_solver_selector.py` 在测试 CSV 上
计算 *regret*：`predicted_time / oracle_time`，并按问题类型切片：

```bash
python3 evaluate_solver_selector.py \
    --csv ../heat_solver_data.csv \
    --model solver_selector.pt
```

输出形如：

```
Top-1 准确率: 0.812
Top-3 准确率: 0.964
Regret (chosen/oracle):
    mean = 1.07     ← 平均比最优只慢 7%
    p90  = 1.21
    max  = 1.93

按问题类型:
  problem      N    top1   regret
  iso         28   0.857    1.04
  aniso       24   0.792    1.08
  conv        24   0.708    1.15
  ...
```

如果 regret 的 mean 接近 1.0，那么即使 Top-1 只有 ~70%，模型仍然非常实用。

---

## 3. C++ 推理：`solver_predictor`

公开 API（`solver_predictor.hpp`）：

```cpp
mfem_ai::SolverPredictor pred("solver_selector.pt");

mfem_ai::HeatProblemDescriptor d;
d.problem = "aniso";   d.mesh_type = "quad";
d.dim = 2;             d.poly_order = 1;
d.n_dof = fes.GetTrueVSize();
d.n_elements = mesh.GetNE();
d.kappa_iso = 33.0;    d.aniso_ratio = 100.0;

mfem_ai::ExtractMatrixFeatures(A_sp, d);   // 自动算 nnz/对称性/SPD…

std::string solver = pred.Predict(d);                  // "PCG_Cheby"
auto ranking      = pred.PredictAll(d);                // [(name, prob)]
```

核心实现要点：

- 通过 PIMPL 把 `<torch/script.h>` 完全藏在 `.cpp` 里 —— 用户代码无需直接依赖 LibTorch 头。
- 类别表与数值特征顺序在 `.cpp` 顶部硬编码，**必须与 Python 训练脚本严格一致**；如有调整请同步两侧。
- 特征向量构造与 `heat_bench.cpp` 完全同口径（同样的 log10、同样的对称性采样），保证训练分布与推理分布对齐。

构建（依赖 LibTorch + 已构建的 MFEM）：

```bash
cd examples/heat_solver_ai
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DMFEM_DIR=<mfem-root> ..
cmake --build . -j
```

运行端到端 demo：

```bash
./heat_solve_with_ai \
    --model solver_selector.pt \
    --problem aniso --mesh-type quad --refine 5 --order 1
```

输出形如：

```
DOF=4225  nnz=33025  sym_ratio=0.000  is_spd=yes

加载模型: solver_selector.pt
推荐排序 (softmax 概率):
  1. PCG_Cheby          0.612
  2. PCG_GS             0.221
  3. CG                 0.119
  ...
→ 选择: PCG_Cheby
[PCG_Cheby] iters=24  time=12.37 ms  converged=yes

=== 验证：前 3 推荐实测耗时 ===
  PCG_Cheby           iters=24    time=12.30 ms  ✓
  PCG_GS              iters=18    time=14.81 ms  ✓
  CG                  iters=85    time=23.92 ms  ✓
```

---

## 4. 集成到自己的 MFEM 程序

最小集成只需链接 `solver_predictor` 这一静态库。库里附带 `RecommendedSolver`
工厂——按预测出的名字一行构造好预条件器和 Krylov 求解器，省去一长串
`if/else if` 派发：

```cpp
#include "solver_predictor.hpp"
using mfem_ai::SolverPredictor;
using mfem_ai::HeatProblemDescriptor;
using mfem_ai::RecommendedSolver;
using mfem_ai::SolverOptions;

// ……组装好 SparseMatrix& A_sp、知道问题语义……
SolverPredictor pred("solver_selector.pt");
HeatProblemDescriptor d;
d.problem = "iso"; d.mesh_type = "hex"; d.dim = 3; d.poly_order = 1;
d.n_dof = fes.GetTrueVSize(); d.n_elements = mesh.GetNE();
mfem_ai::ExtractMatrixFeatures(A_sp, d);

SolverOptions opts;     // rtol=1e-8, atol=1e-12, max_iter=3000, kdim=50
RecommendedSolver rs(pred.Predict(d), A_sp, opts);
rs.Solve(B, x);

std::cout << rs.Name() << " iters=" << rs.NumIterations()
          << " converged=" << rs.Converged() << "\n";
```

如果想自己处理求解器实例化，只调 `pred.Predict(d)` / `pred.PredictAll(d)`
即可，工厂部分完全可选。

---

## 5. 维护：增删求解器/特征

任何一边改了，**两边必须同步**：

| 改动 | Python 端 | C++ 端 |
| ---- | ---------- | ------- |
| 增加求解器 | `SOLVERS` 列表 | `kSolvers`，并在 `RunSolver`/`Solve` 增加分支 |
| 增加问题   | `PROBLEMS` 列表 + 特征工程支持 | `kProblems`，benchmark + demo 加分支 |
| 增加数值特征 | `NUMERIC_FEATURES` + `engineer_features` | `kNumericNames` + `BuildFeatureVector` |

不同步时模型加载会失败（特征维度不一致 → 抛 `runtime_error`）；修改后必须重训练并重新导出 TorchScript。
