"""
TableShift 外部验证：在**同一划分、同一特征预算**下比较三类特征选择对 ID/OOD 的影响。

三类方法
--------
mi_topk    普通特征选择：训练集上逐特征互信息，取前 k
iamb_pure  纯数据因果特征选择：IAMB 求标签的马尔可夫毯，再按预算截断/补齐
iamb_soft  同一 IAMB + 语言模型先验作为候选排序信号（gamma 与 bnlearn 主实验一致）

为什么三种方法必须选同样多的特征：否则"因果方法更好"可能只是"多给了几个特征"。
所以 IAMB 的毯子先按互信息排序，不足 k 用下一个最高互信息的特征补齐，超过 k 截断。

数据划分
--------
直接使用 TableShift 官方划分导出的 CSV：train / validation / id_test / ood_test。
编码、缺失填补、分箱全部只在 train 上拟合。

用法
----
    python scripts/tableshift/run_tableshift.py --data data/tableshift --out results
    python scripts/tableshift/run_tableshift.py --self-test        # 合成数据自检

先验文件（可选）
----------------
data/tableshift/<task>/prior_role.csv，两列：feature,score
score ∈ [-1, 1]：正=倾向在标签的马尔可夫毯内，负=倾向不在。缺省时只跑前两种方法，
并在结果里标 method_available=0，不假装跑过。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mb.ci_test import CITest  # noqa: E402
from src.mb.discovery import iamb  # noqa: E402

SPLITS = ("train", "validation", "id_test", "ood_test")
N_BINS = 5          # 连续特征做条件独立检验前的分箱数，只在 train 上定边界


# ----------------------------------------------------------------------
# 读数据与编码
# ----------------------------------------------------------------------

def load_task(task_dir: Path):
    frames = {}
    for split in SPLITS:
        path = task_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"缺少划分文件：{path}")
        frames[split] = pd.read_csv(path)
    return frames


def split_xy(frame: pd.DataFrame, target: str = "target"):
    if target not in frame.columns:
        raise KeyError(f"找不到目标列 {target}，现有列：{list(frame.columns)[:10]}")
    y = frame[target].astype(int).to_numpy()
    x = frame.drop(columns=[target])
    return x, y


def fit_encoder(train_x: pd.DataFrame, n_bins: int = N_BINS):
    """在训练集上拟合：连续列的分箱边界、类别列的取值表。"""

    spec = {}
    for col in train_x.columns:
        s = train_x[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > n_bins:
            edges = np.unique(np.nanquantile(s.to_numpy(dtype=float), np.linspace(0, 1, n_bins + 1)[1:-1]))
            spec[col] = ("num", edges)
        elif pd.api.types.is_numeric_dtype(s):
            spec[col] = ("num_small", float(s.median()) if s.notna().any() else 0.0)
        else:
            values = s.astype("string").fillna("__NA__").value_counts().index.tolist()
            spec[col] = ("cat", {v: i for i, v in enumerate(values)})
    return spec


def transform(x: pd.DataFrame, spec, for_ci: bool = False):
    """返回 float 矩阵。for_ci=True 时连续列换成箱号，供 G 检验使用。"""

    out = np.zeros((len(x), len(spec)), dtype=np.float64)
    for j, (col, (kind, param)) in enumerate(spec.items()):
        s = x[col]
        if kind == "num":
            values = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            if for_ci:
                codes = np.digitize(values, param)
                codes = np.where(np.isnan(values), -1, codes)
                out[:, j] = codes
            else:
                out[:, j] = values
        elif kind == "num_small":
            out[:, j] = pd.to_numeric(s, errors="coerce").fillna(param).to_numpy()
        else:
            out[:, j] = s.astype("string").fillna("__NA__").map(param).fillna(-1).to_numpy()
    return out


def discrete_mask(spec):
    return np.array([spec[c][0] in ("cat", "num_small") for c in spec])


# ----------------------------------------------------------------------
# 三种选择
# ----------------------------------------------------------------------

def mi_order(matrix_ci: np.ndarray, y: np.ndarray, mask, seed: int):
    scores = mutual_info_classif(
        matrix_ci, y, discrete_features=mask, random_state=seed
    )
    return np.argsort(-scores, kind="stable"), scores


def equalize(mb_indices, mi_rank, k: int):
    """把马尔可夫毯整理成恰好 k 个特征：先按互信息排序，再补齐/截断。"""

    chosen = [int(v) for v in mi_rank if int(v) in set(int(m) for m in mb_indices)]
    for v in mi_rank:
        if len(chosen) >= k:
            break
        if int(v) not in chosen:
            chosen.append(int(v))
    return chosen[:k]


def iamb_mb(matrix_ci: np.ndarray, y: np.ndarray, prior=None, gamma: float = 0.0,
            max_k: int = 3):
    n_features = matrix_ci.shape[1]
    data = np.column_stack([matrix_ci.astype(np.int32), y.astype(np.int32)])
    ci = CITest(data, alpha=0.05)
    mb = iamb(ci, target=n_features, n_vars=n_features + 1, max_k=max_k,
              prior=prior, gamma=gamma, normalize=True)
    return sorted(int(v) for v in mb if int(v) != n_features)


def load_prior(task_dir: Path, columns):
    path = task_dir / "prior_role.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    index = {name: i for i, name in enumerate(columns)}
    scores = {}
    target_index = len(columns)
    for _, row in table.iterrows():
        name = str(row["feature"])
        if name not in index:
            continue
        value = float(row["score"])
        value = max(-1.0, min(1.0, value))
        i = index[name]
        scores[(min(i, target_index), max(i, target_index))] = value
    return scores


# ----------------------------------------------------------------------
# 下游模型与评测
# ----------------------------------------------------------------------

def fit_predict(train_x, y_train, test_sets, seed: int):
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=None,
        min_samples_leaf=40, l2_regularization=0.1, random_state=seed,
    )
    model.fit(train_x, y_train)
    return {name: model.predict_proba(x)[:, 1] for name, (x, _) in test_sets.items()}, model


def score(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def run_task(task_dir: Path, budget: int, seeds, gamma: float, max_k: int):
    frames = load_task(task_dir)
    x_train, y_train = split_xy(frames["train"])
    spec = fit_encoder(x_train)
    mask = discrete_mask(spec)

    m_train = transform(x_train, spec)
    m_train_ci = transform(x_train, spec, for_ci=True)

    k = int(min(budget, len(spec) - 1))
    rank, _ = mi_order(m_train_ci, y_train, mask, seed=seeds[0])

    selections = {}
    selections["mi_topk"] = [int(v) for v in rank[:k]]
    selections["iamb_pure"] = equalize(iamb_mb(m_train_ci, y_train, None, 0.0, max_k), rank, k)

    prior = load_prior(task_dir, list(spec))
    if prior:
        selections[f"iamb_soft_g{gamma:g}"] = equalize(
            iamb_mb(m_train_ci, y_train, prior, gamma, max_k), rank, k
        )

    rows = []
    columns = list(spec)
    for method, chosen in selections.items():
        cols = [columns[i] for i in chosen]
        train_sel = x_train[cols]
        test_sets = {}
        for split in ("validation", "id_test", "ood_test"):
            xs, ys = split_xy(frames[split])
            test_sets[split] = (xs[cols], ys)

        for seed in seeds:
            preds, _ = fit_predict(train_sel, y_train, test_sets, seed)
            m_val = score(test_sets["validation"][1], preds["validation"])
            m_id = score(test_sets["id_test"][1], preds["id_test"])
            m_ood = score(test_sets["ood_test"][1], preds["ood_test"])
            rows.append({
                "task": task_dir.name, "method": method, "seed": seed, "k": k,
                "selected": ";".join(cols),
                "auc_val": m_val["auc"], "auc_id": m_id["auc"], "auc_ood": m_ood["auc"],
                "gap_id_ood": m_id["auc"] - m_ood["auc"],
                "acc_id": m_id["acc"], "acc_ood": m_ood["acc"],
                "f1_id": m_id["f1"], "f1_ood": m_ood["f1"],
            })
    return rows


# ----------------------------------------------------------------------
# 合成数据自检
# ----------------------------------------------------------------------

def make_synthetic(root: Path, n_train=1200, n_test=400, seed=0):
    rng = np.random.default_rng(seed)
    task = root / "synthetic_shift"
    task.mkdir(parents=True, exist_ok=True)

    def draw(n, shift=0.0):
        x0 = rng.normal(shift, 1.0, n)
        x1 = rng.normal(0, 1.0, n)
        x2 = rng.normal(0, 1.0, n)
        x3 = rng.normal(0, 1.0, n)
        cat = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
        logit = 1.6 * x0 - 1.2 * x1 + 0.8 * x2 + 0.4 * (cat == "b") - 0.3
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        return pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "x3": x3, "cat": cat, "target": y})

    draw(n_train).to_csv(task / "train.csv", index=False)
    draw(n_test).to_csv(task / "validation.csv", index=False)
    draw(n_test).to_csv(task / "id_test.csv", index=False)
    draw(n_test, shift=1.2).to_csv(task / "ood_test.csv", index=False)

    # 合成先验：真相关的是 x0/x1/x2，x3 与 cat 判为无关
    pd.DataFrame({
        "feature": ["x0", "x1", "x2", "x3", "cat"],
        "score": [0.9, 0.8, 0.7, -0.6, -0.8],
    }).to_csv(task / "prior_role.csv", index=False)
    (task / "schema.json").write_text(json.dumps({"task": "synthetic_shift"}), encoding="utf-8")
    return task


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tableshift")
    parser.add_argument("--out", default="results")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--tag", default="", help="输出文件名后缀，如 _g4")
    args = parser.parse_args()

    if args.self_test:
        tmp = Path(tempfile.mkdtemp(prefix="ts_selftest_"))
        make_synthetic(tmp)
        rows = run_task(tmp / "synthetic_shift", budget=3, seeds=[7], gamma=2.0, max_k=2)
        df = pd.DataFrame(rows)
        print(df[["task", "method", "k", "auc_id", "auc_ood", "gap_id_ood", "selected"]]
              .to_string(index=False))
        assert {"mi_topk", "iamb_pure"} <= set(df["method"]), "前两种方法没有全部运行"
        assert any(m.startswith("iamb_soft") for m in df["method"]), "先验方法没有运行"
        assert df["auc_id"].notna().all(), "ID 评测出现 NaN"
        print("自检通过：三种方法都产出 ID/OOD 指标")
        return 0

    data_root = Path(args.data)
    if not data_root.exists():
        raise SystemExit(f"找不到数据目录：{data_root}（先在 tableshift 环境里跑 fetch_tableshift.py）")

    tasks = ([t for t in args.tasks.split(",") if t] if args.tasks
             else sorted(p.name for p in data_root.iterdir() if p.is_dir()))

    rows = []
    for name in tasks:
        task_dir = data_root / name
        try:
            rows.extend(run_task(task_dir, args.budget, args.seeds, args.gamma, args.max_k))
        except Exception as error:                       # noqa: BLE001
            print(f"[{name}] 失败：{type(error).__name__}: {error}")
            continue

    if not rows:
        raise SystemExit("没有任何任务跑成功")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = out_dir / f"tableshift_metrics{args.tag}.csv"
    pd.DataFrame(rows).to_csv(detail, index=False)

    df = pd.DataFrame(rows)
    summary = (df.groupby(["task", "method"])
                 .agg(auc_id=("auc_id", "mean"), auc_ood=("auc_ood", "mean"),
                      gap=("gap_id_ood", "mean"), gap_sd=("gap_id_ood", "std"),
                      k=("k", "first"), n=("seed", "count"))
                 .reset_index())
    summary.to_csv(out_dir / f"tableshift_summary{args.tag}.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\n-> {detail}\n-> {out_dir / ('tableshift_summary' + args.tag + '.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
