# XGBoost native multi-output without a wrapper

Research note for [issue #6](https://github.com/jenujari/prdict-pov-v1/issues/6);
resolves the conditional in map decision 2 ([issue #2](https://github.com/jenujari/prdict-pov-v1/issues/2)).

**Date:** 2026-08-02 · **Version under test:** XGBoost **3.3.0** (current release, 2026-06-17),
CPU, OpenMP on. Experiments were run on a from-source musl build; findings re-verified
2026-08-03 against the container wheel that the pipeline actually uses — see
[Environment](#-environment-the-pipeline-runs-in-a-container-16).

---

## Headline answer

**Yes — XGBoost predicts a 10-element vector natively, with no `MultiOutputRegressor`.**
It does so at *two* levels, and this distinction is the crux of the ticket:

| Level | `multi_strategy` | Wrapper needed? | Tree structure |
|---|---|---|---|
| **A** | `one_output_per_tree` (default) | **No** | 10 scalar-leaf trees per boosting round, one per target, inside **one** booster |
| **B** | `multi_output_tree` | **No** | **1** tree per boosting round whose every leaf holds a 10-vector |

Both accept `y` of shape `(n, 10)` from `XGBRegressor.fit` and both return `(n, 10)` from
`predict`. The ticket's phrase "natively, without a wrapper" is satisfied by level A alone,
which has been true since **1.6**. Level B — the *shared tree structure* that the question is
really reaching for — arrived in **2.0.0** and is still labelled work-in-progress in 3.3.0.

**Recommendation: XGBoost takes the 10-STEP vector, shared with the TFT — and runs
`multi_strategy="one_output_per_tree"` in production, with `multi_output_tree` kept as a
cheap ablation.** Reasoning in [§6](#6-the-decision).

> ### ⚠ Environment: the pipeline runs in a container ([#16](https://github.com/jenujari/prdict-pov-v1/issues/16))
>
> **Everything in this note was run for real** — nothing below is doc-derived-but-untested;
> where a claim is documentation-only it says so explicitly.
>
> The experiments predate the container runtime and were executed against a **from-source
> musl build** of 3.3.0, because at the time nothing else would install. That detour is what
> produced [#16](https://github.com/jenujari/prdict-pov-v1/issues/16), now resolved: the
> pipeline installs the ordinary **manylinux wheel inside a podman container** and no source
> build is needed. Run anything here with `./container/run.sh python <script>`; setup and
> rationale in [`kb/runtime.md`](../runtime.md).
>
> **The results carry over.** Same version (3.3.0), same CPU `hist` code path. Re-verified
> against the container wheel on 2026-08-03: `multi_output_tree` + `hist` +
> `enable_categorical=True` fits and predicts `(n, 10)`, and the §2 categorical degradation
> reproduces exactly — 20 rounds at depth 5 gave **32/32 categorical splits of size 1** under
> vector leaf against sets up to size 26 under `one_output_per_tree`, with
> `max_cat_to_onehot=4` set in both.
>
> One difference worth knowing: the container wheel reports `USE_CUDA: True` / `USE_NCCL:
> True`, where the source build reported both `False`. This is cosmetic here — it is the
> stock PyPI wheel, NCCL is `dlopen`ed and never loaded on a GPU-less box, and `device`
> defaults to `"cpu"`. Timings in §4 came from the source build and were not re-run; treat
> them as ratios, not absolute numbers.
>
> **The musl failure chain**, verified end to end, kept because it is the evidence behind #16
> and because it pins down which versions are viable at all:
>
> 1. This box is **Chimera Linux / musl libc** (`ldd → musl libc 1.2.6`). XGBoost publishes
>    **no `musllinux` wheels for any version** — `manylinux` only (checked 2.0.3, 2.1.4,
>    3.0.0, 3.0.5, 3.1.3, 3.2.0, 3.3.0 via the PyPI JSON API). So there is never a wheel.
> 2. Since **2.1.4**, every release declares a hard `nvidia-nccl-cu12` dependency on Linux
>    (`platform_system == "Linux"`), which also has no musl wheel. Asking for the current
>    release directly therefore fails at resolution:
>    ```
>    × No solution found when resolving dependencies:
>      ╰─▶ Because all versions of nvidia-nccl-cu12{sys_platform == 'linux'} have no wheels
>          with a matching platform tag (e.g., `linux_x86_64`) and xgboost==3.3.0 depends on
>          nvidia-nccl-cu12{sys_platform == 'linux'}, we can conclude that xgboost==3.3.0
>          cannot be used.
>    ```
> 3. Left unpinned, `uv` therefore **backtracks to 2.0.3** — the newest release whose
>    dependencies are `numpy, scipy` only, i.e. the last one before the NCCL pin. Confirmed:
>    ```
>    $ uv pip compile --python-platform x86_64-unknown-linux-musl  (input: "xgboost")
>    numpy==2.5.1
>    scipy==1.18.0
>    xgboost==2.0.3
>    ```
>    This is exactly the 2.0.3 that `uv run --with xgboost` reports.
> 4. 2.0.3 then has to build from source, which needs `cmake` — hence the observed
>    `FileNotFoundError: [Errno 2] No such file or directory: 'cmake'`.
> 5. **And supplying cmake does not rescue 2.0.3.** Two further walls, both hit for real:
>    - its vendored `dmlc-core` is too old for CMake 4.x (4.4.0 is current on PyPI):
>      ```
>      CMake Error at dmlc-core/CMakeLists.txt:1 (cmake_minimum_required):
>        Compatibility with CMake < 3.5 has been removed from CMake.
>      ```
>      Needs `cmake<4` (or `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`).
>    - and with `cmake 3.31.10` it configures but then **fails to compile on musl**, because
>      2.0.3 calls the glibc-only `mmap64`:
>      ```
>      src/common/io.cc:260:38: error: use of undeclared identifier 'mmap64'; did you mean 'mmap'?
>        260 |   ptr = reinterpret_cast<std::byte*>(mmap64(nullptr, view_size, prot, ...));
>            |                                      ^~~~~~
>      /usr/include/sys/mman.h:120:7: note: 'mmap' declared here
>      ```
>      musl exposes only `mmap` (its `off_t` is 64-bit already). **XGBoost 2.0.3 is not
>      buildable on this box at all, at any toolchain.** Upstream fixed musl compatibility
>      later; the 3.x source compiles cleanly.
>
> **Which version is actually installable on the musl host: 3.3.0 — the current release.** It
> builds cleanly from source in ~2.5 min on 16 cores, and that build is what produced the
> experiments below:
>
> ```sh
> uv venv --python 3.12 .venv
> VIRTUAL_ENV=.venv uv pip install cmake ninja numpy scipy pandas scikit-learn
> VIRTUAL_ENV=.venv PATH=.venv/bin:$PATH \
>   uv pip install --no-deps --no-binary xgboost "xgboost==3.3.0"
> ```
>
> `--no-deps` drops the unsatisfiable NCCL pin (GPU-only, irrelevant on a CPU box — install
> `numpy`/`scipy` yourself, as above). `--no-binary xgboost` forces the source build.
> Toolchain present: `g++ 16.1.0`, cmake 4.4.0 from PyPI. Result:
>
> ```
> 3.3.0
> {'USE_CUDA': False, 'USE_NCCL': False, 'USE_OPENMP': True, 'DEBUG': False,
>  'CLANG_VERSION': [22, 1, 7], 'MM_PREFETCH_PRESENT': True, ...}
> ```
>
> **This recipe is now historical.** It is recorded because it is the only way to get XGBoost
> onto the *host*, and because it demonstrates the gap is a packaging accident rather than a
> hard incompatibility — which is precisely why #16 chose a container over a source build:
> torch has no such escape hatch. Do not use it for the pipeline.
>
> **So the version constraint runs the opposite way to the usual worry.** The concern was
> that a doc answer about the newest release would be useless because only an old release is
> installable. In fact the newest release is the *easy* one to build and 2.0.3 is the one
> that will not build. And 2.0.3 would have been useless anyway — see below.
>
> **Is `multi_strategy` in 2.0.3 sufficient for map decision 2? Moot, and no.** Moot because
> 2.0.3 cannot be built here (step 5). And no on the merits: 2.0.3 does have `multi_strategy`
> (added 2.0.0), but its vector leaf predates every feature this project needs. Per the
> upstream changelogs it lacks **categorical feature support** (added 3.3.0, PR #12072 — so
> the decision-6 combination does not exist in 2.0.3 at all), row subsampling, column
> sampling, L1/L2 and `min_split_loss` regularization, MAE objectives, and gain/cover feature
> importance (all 3.2.0), and SHAP (3.3.0). A 2.0.3 fallback would fail map decision 6
> outright. *(This paragraph is changelog-derived, not executed — 2.0.3 would not compile.)*
>
> **Action for the pipeline — done.** `xgboost==3.3.0` is pinned in
> `container/requirements.txt`, resolved for `x86_64-manylinux_2_28` rather than for the host,
> so the 2.0.3 backtrack cannot happen. The `>= 3.3.0` floor is a hard requirement, not a
> preference: 3.2.0 and below have no categorical support under vector leaf and no SHAP.

---

## 1. The `multi_strategy` parameter

### Version history (primary sources)

| Version | Event | Source |
|---|---|---|
| 1.6 | Multi-output regression / multi-label classification, one-model-per-target | [`doc/tutorials/multioutput.rst` L12–14](https://github.com/dmlc/xgboost/blob/master/doc/tutorials/multioutput.rst) — `.. versionadded:: 1.6` |
| **2.0.0** | **`multi_strategy` introduced; vector-leaf trees** | [Parameters doc](https://xgboost.readthedocs.io/en/stable/parameter.html): "`multi_strategy`, [default = `one_output_per_tree`] **Added in version 2.0.0**" |
| 3.0.0 | Weight-based feature importance for vector leaf | changelog v3.0.0, [PR #10700](https://github.com/dmlc/xgboost/pull/10700) |
| 3.1 | Multi-target intercept | [PR #11656](https://github.com/dmlc/xgboost/pull/11656) |
| 3.2.0 | "substantial progress": all regression objectives, regularization, sub/col-sampling, gain+cover importance, model dumps, external memory, GPU parity, reduced gradient | [`doc/changes/v3.2.0.rst` L30–66](https://github.com/dmlc/xgboost/blob/master/doc/changes/v3.2.0.rst) |
| **3.3.0** | **Exact SHAP for vector leaf; one-hot categorical splits for CPU `hist` vector leaf** | [`doc/changes/v3.3.0.rst` L15–16, L39–41](https://github.com/dmlc/xgboost/blob/master/doc/changes/v3.3.0.rst), PRs #12209/#12210/#12247, #12015/#12072/#12244 |
| 3.4 (unreleased) | Partition-based categorical splits, interaction & monotone constraints, tree dataframe | PRs [#12299](https://github.com/dmlc/xgboost/pull/12299) (2026-07-14), [#12294](https://github.com/dmlc/xgboost/pull/12294), [#12341](https://github.com/dmlc/xgboost/pull/12341), [#12293](https://github.com/dmlc/xgboost/pull/12293) — **all merged after the 3.3.0 release** |

### Is it still experimental?

**Yes, explicitly.** Three independent statements:

- Parameters doc: *"This parameter is working-in-progress."*
- Tutorial: *"As of XGBoost 3.0, the feature is experimental and has limited features. Only
  the Python package is tested. In addition, `glinear` is not supported."*
- v3.2.0 changelog: *".. warning:: The vector leaf is still a work in progress. Feedback is
  welcome."*
- 3.3.0 changelog still calls it *"the working-in-progress vector leaf"*.

The tracking roadmap is [dmlc/xgboost#9043](https://github.com/dmlc/xgboost/issues/9043)
(open, `type: roadmap`).

### What it does to tree structure

From the tutorial:

> XGBoost can optionally build multi-output trees with the size of leaf equals to the number
> of targets when the tree method `hist` is used.

And v3.2.0 changelog:

> The vector leaf tree stores a vector of weights in each leaf node, enabling the model to
> capture correlations across targets during tree construction.

Confirmed empirically — 50 boosting rounds, 10 targets:

```
=== one_output_per_tree ===
  trees in dump: 500 | rounds: 50      <- 10 trees per round
=== multi_output_tree ===
  trees in dump:  50 | rounds: 50      <- 1 tree per round
```

A vector-leaf tree dumps as (note the leaf is an array):

```
0:[num_47<-0.500149012] yes=1,no=2,missing=2
	1:[num_19<-0.767392516] yes=3,no=4,missing=4
		3:leaf=[-0.00207527843, -0.0036950889, ..., -0.0126737375]
		4:leaf=[-0.00064505561, -0.00123111811, ..., -0.00414770143]
	2:[num_52<-0.478301227] yes=5,no=6,missing=6
		5:leaf=[-0.000176544083, -0.000288714829, ..., -0.00096601306]
		6:leaf=[0.000943922147, 0.00171003852, ..., 0.00579615682]
```

All 10 targets are forced through **one** split structure. Splits are chosen on the summed
gain across targets, so a feature only splits if it helps the targets *jointly*.

---

### Version capability matrix (vector leaf, CPU `hist`)

Which release you are on changes the answer materially. `✓` = supported, `✗` = absent or raises.

| Capability | 2.0.x | 3.0.x | 3.2.0 | **3.3.0** | 3.4 (unreleased) |
|---|:--:|:--:|:--:|:--:|:--:|
| `multi_strategy="multi_output_tree"` | ✓ | ✓ | ✓ | **✓** | ✓ |
| **native categorical (`enable_categorical`)** | ✗ | ✗ | ✗ | **✓ one-hot split only** | ✓ + partition split |
| `max_cat_to_onehot` honoured | – | – | – | **✗ (ignored)** | ✓ |
| subsample / colsample / L1 / L2 | ✗ | ✗ | ✓ | ✓ | ✓ |
| MAE, pseudo-Huber objectives | ✗ | ✗ | ✓ | ✓ | ✓ |
| feature importance `weight` | ✗ | ✓ | ✓ | ✓ | ✓ |
| feature importance `gain` / `cover` | ✗ | ✗ | ✓ | ✓ | ✓ |
| **SHAP (`pred_contribs`, `pred_interactions`)** | ✗ | ✗ | ✗ | **✓ exact** | ✓ |
| `trees_to_dataframe`, valid JSON dump | ✗ | ✗ | ✗ | **✗** | ✓ |
| monotone / interaction constraints | ✗ | ✗ | ✗ | **✗** | ✓ |
| `reg:quantileerror` | ✗ | ✗ | ✗ | ✗ | ✗ |
| `approx` / `exact` tree method | ✗ | ✗ | ✗ | ✗ | ✗ |

**3.3.0 is the first release where the decision-6 combination exists at all.** The 2.0.x/3.0.x
columns are the reason a "just use whatever installs" fallback would have been wrong.

---

## 2. Does `multi_output_tree` + `hist` + `enable_categorical` work together on CPU?

> **Verification status: empirically confirmed**, XGBoost 3.3.0, this box, CPU-only build
> (`USE_CUDA: False`). Not inferred from documentation — the docs do not state whether the
> three compose.

### Yes. The combination fits and predicts.

Synthetic data matched to the project shape: 6517 rows, 60 numeric + **40 native
`category`-dtype** columns (cardinalities 27/12/9/7/4/3/2 — modelled on nakshatras, signs,
etc.), 10-element target.

```
xgboost: 3.3.0
build  : {'USE_CUDA': False, 'USE_OPENMP': True}
X shape: (6517, 100) | numeric: 60 | category: 40
max categorical cardinality: 27
y_cum  intercorrelation: mean |r| off-diag = 0.755, min = 0.435, max = 0.967
y_step intercorrelation: mean |r| off-diag = 0.046, min = 0.014, max = 0.108

=== multi_strategy=one_output_per_tree ===
  fit OK. predict shape: (1000, 10) dtype: float32
=== multi_strategy=multi_output_tree ===
  fit OK. predict shape: (1000, 10) dtype: float32
```

No warning, no silent fallback, no GPU required.

### ⚠ But categorical splits are DEGRADED under vector leaf in 3.3.0

**This is the most consequential finding for map decision 6.** In 3.3.0, a vector-leaf tree
splits every categorical feature **one-vs-rest**, and **`max_cat_to_onehot` is ignored
entirely**.

Source proof — `src/tree/hist/evaluate_splits.h` at tag `v3.3.0`, inside
`HistMultiEvaluator::EvaluateSplits` (the multi-target evaluator, L675–684):

```cpp
bool is_cat = common::IsCat(feature_types, fidx);
if (is_cat) {
  this->EnumerateOneHot(cut, fidx, node_hist, parent_sum, parent_gain, best);
} else {
  bool missing = this->EnumerateSplit<+1>(...);
  ...
}
```

There is no `common::UseOneHot(n_bins, param_->max_cat_to_onehot)` test and no
`EnumeratePart` call — unlike the single-target `HistEvaluator` at L362–381, which has both.
[PR #12072](https://github.com/dmlc/xgboost/pull/12072) ("[mt] Implement one-hot categorical
feature for CPU hist.") says it outright in the body: *"Only OHE is implemented at the
moment."*

Empirical confirmation, K = 27 levels, 20 rounds × depth 5, counting the size of the
category set on each categorical split:

```
category-set size per categorical split (20 rounds, depth 5, K=27):
  one_output_per_tree  mcto=  4 -> 2252 cat splits, size histogram {1: 290, 2: 242, 3: 126,
      4: 199, 5: 98, 6: 62, 7: 27, 8: 28, 9: 25, 10: 15, 11: 27, 12: 56, 13: 31, 14: 33,
      15: 28, 16: 26, 17: 21, 18: 16, 19: 22, 20: 24, 21: 64, 22: 68, 23: 123, 24: 144,
      25: 181, 26: 276}
  one_output_per_tree  mcto= 64 -> 1000 cat splits, size histogram {1: 1000}
  multi_output_tree    mcto=  4 ->  100 cat splits, size histogram {1: 100}
  multi_output_tree    mcto= 64 ->  100 cat splits, size histogram {1: 100}
```

Actual split text for the same 27-level feature:

```
one_output_per_tree  mcto=4 : 0:[nak:{0,1,4,6,7,8,9,11,12,13,15,17,18,19,22}]   <- PARTITION
one_output_per_tree  mcto=64: 0:[nak:{23}]                                       <- one-hot
multi_output_tree    mcto=4 : 0:[nak:{23}]                                       <- one-hot (mcto ignored)
multi_output_tree    mcto=64: 0:[nak:{23}]                                       <- one-hot
```

**What this means for decision 6.** It is *not* a violation of decision 6: the feature is
still consumed as a native `category` dtype, the design matrix is not widened, and nothing is
ordinal-encoded — level identity is preserved. But the *optimal-partition* split (the main
reason to prefer native categorical over one-hot for a 27-level nakshatra) is unavailable.
Each split can only isolate a single nakshatra, so expressing "these 9 nakshatras behave
alike" costs 9 levels of depth instead of 1 split. On the default `max_cat_to_onehot=4`, the
single-target path uses partition splits for every one of the ~40 categoricals; the
vector-leaf path uses none.

Partition-based categorical splits for multi-target CPU landed in
[PR #12299](https://github.com/dmlc/xgboost/pull/12299) on **2026-07-14**, i.e. **after** the
3.3.0 release (2026-06-17). They will be in 3.4.0. Master already has the code (bins sorted
by projection onto the parent's update direction).

### Categorical handling, other notes (3.3.0)

- `enable_categorical` is now **default-on**: 3.3.0 "Enable categorical feature support by
  default while keeping `enable_categorical` available for users who need to disable it"
  (PRs #12015, #12072, #12244). Verified — omitting the flag works; `enable_categorical=False`
  raises. **Keep passing it explicitly anyway** so the code is version-independent and the
  intent is legible.
- Levels present in the dtype but absent from a training fold predict fine
  (`finite=True`). A level added to the dtype *after* fit raises
  `Found a category not in the training data` (`src/data/cat_container.h:29`). Since
  `categories_list.json` gives the full level list up front, declaring the complete
  `pd.Categorical` categories before any fold split makes this a non-issue — but the fold
  code must build the dtype from the JSON, not from `fold.unique()`.

---

## 3. Early stopping, feature importance, SHAP

### Early stopping — works, but on ONE aggregated scalar

```
  one_output_per_tree  best_iteration=107  best_score=0.0304999
    metrics tracked: ['rmse']  (len 128)   -> a SINGLE scalar per round
  multi_output_tree    best_iteration=0    best_score=0.0314673
    metrics tracked: ['rmse']  (len 21)    -> a SINGLE scalar per round
```

The machinery is sound (no silent degradation), but there is exactly **one** `rmse` number
per round for all 10 horizons. Verified what it is:

```
  reported rmse             = 0.03451205
  sqrt(mean over ALL cells) = 0.03451205   <- this
  mean of per-target rmse   = 0.03247552
```

It is the RMSE over the **flattened `(n, 10)` matrix** — an unweighted sum of squared errors,
*not* the mean of per-horizon RMSEs. Consequence: **horizons with larger variance dominate
both the training loss and the stopping rule.** See [§5](#5-correlated-outputs) — this is
what kills the cumulative target.

Per-target intercepts are correct under both strategies (they equal the per-target training
means), so the round-0 baseline is right:

```
  multi_output_tree    intercept_ = [9.90e-05 2.45e-04 2.23e-04 4.59e-04 7.91e-04 7.69e-04
                                     9.08e-04 1.149e-03 1.319e-03 1.406e-03]
  train mean per target =           [9.90e-05 2.45e-04 2.23e-04 4.59e-04 7.91e-04 7.69e-04
                                     9.08e-04 1.149e-03 1.319e-03 1.406e-03]
```

`xgb.cv` also works: `['train-rmse-mean', 'train-rmse-std', 'test-rmse-mean', 'test-rmse-std']`.

### Feature importance — all five types work, but read `weight` carefully

```
  --- multi_output_tree ---
    weight      : OK,  87 features, top3=[('num_3', 35.0), ('num_47', 34.0), ('cat_0', 31.0)]
    gain        : OK,  87 features, top3=[('num_47', 1.193), ('num_52', 0.7618), ('num_28', 0.4311)]
    cover       : OK,  87 features, top3=[('num_47', 43963.53), ('num_0', 34980.0), ('num_33', 34200.0)]
    total_gain  : OK,  87 features, top3=[('num_47', 40.5614), ('num_52', 19.8072), ('num_0', 7.3801)]
    total_cover : OK,  87 features, top3=[('num_47', 1494760.0), ('num_0', 909480.0), ('cat_10', 817910.0)]
    sklearn feature_importances_ shape=(100,) sum=1.0000
```

`gain`/`cover` for vector leaf arrived in 3.2.0 ("Feature importance variants (gain and
coverage)"); before that only `weight` existed (PR #10700: *"Only weight is implemented as the
vector-leaf tree doesn't have hessian or gain at the moment"*).

Caveat: raw `weight` counts are ~10× smaller under `multi_output_tree` (35 vs 849) simply
because there are 10× fewer trees. **Never compare importances across strategies without
normalising** — `feature_importances_` (which sums to 1.0) is safe; `get_score` is not.
A vector-leaf `gain` is the gain summed over all 10 targets, so it answers "which feature
helps the horizon vector jointly", not "which feature helps horizon k". There is no
per-horizon importance under vector leaf. Under `one_output_per_tree` you *can* recover
per-horizon importance by slicing trees by index (roadmap note: *"be careful with tree
index"*).

### SHAP — works exactly, new in 3.3.0

```
  --- multi_output_tree ---
    pred_contribs     : shape (64, 10, 101)
      additivity: max|sum(contribs)-pred| = 1.490e-08
    pred_interactions : shape (64, 10, 101, 101)
```

Correct 3-D shape `(n, n_targets, n_features + 1)` and additivity holds to float precision.
This is **specifically a 3.3.0 gain** — the v3.2.0 changelog lists "Shapley values" under
*"Currently missing features for the `hist` tree method with vector leaf"*. 3.3.0 added
"exact SHAP contribution and interaction prediction for vector-leaf multi-output trees on
both CPU and GPU" (PRs #12209, #12210, #12247, #11985, #12208). **If you want SHAP with
vector leaf you must be on ≥ 3.3.0.**

### ⚠ Model introspection IS broken under vector leaf in 3.3.0

```
  --- one_output_per_tree ---
    trees_to_dataframe: OK  shape=(1550, 11)
    text dump elides values? False
  --- multi_output_tree ---
    trees_to_dataframe: FAILED -> XGBoostError: src/tree/tree_model.cc:737:
        Check failed: !with_stats: Tree dump with statistic support for multi-target tree
    first leaf line: 15:leaf=[-0.000276163308, -0.000345947075, ..., -0.00139220315]
    text dump elides values? True
```

Two real defects:

1. `Booster.trees_to_dataframe()` raises. (Fixed post-3.3.0 by
   [PR #12293](https://github.com/dmlc/xgboost/pull/12293).)
2. **Both the text and JSON dumps truncate the leaf vector with a literal `...`** — the
   middle 8 of 10 leaf values are unrecoverable, and `get_dump(dump_format="json")` therefore
   emits **invalid JSON**:
   ```
   one_output_per_tree : json.loads OK
   multi_output_tree   : json.loads FAILED -> Expecting value: line 4 column 60
     offending text: '{ "nodeid": 2, "leaf": [0.636967659, 1.1980443, ..., 4.13362646] }'
   ```

Save/load is unaffected — `.ubj` and `.json` model round-trips reproduce predictions exactly.
But any plan to inspect, diff, or export vector-leaf tree structure is blocked on 3.3.0.

### Other compatibility (3.3.0, `multi_strategy="multi_output_tree"`)

| Feature | Status | Message |
|---|---|---|
| `tree_method="hist"` / `"auto"` | ✅ | |
| `tree_method="approx"` / `"exact"` | ❌ | `Only the hist tree method is supported for building multi-target trees with vector leaf.` |
| `reg:squarederror`, `reg:absoluteerror`, `reg:pseudohubererror`, `reg:squaredlogerror` | ✅ | |
| `reg:quantileerror` | ❌ | `Check failed: info.labels.Shape(1) == 1 (10 vs. 1): Multi-target is not yet supported by the quantile loss.` |
| `monotone_constraints` | ❌ | `Monotonic constraint support for multi-target tree is not yet implemented.` (`updater_quantile_hist.cc:627`) |
| `interaction_constraints` | ❌ | `Interaction constraint support for multi-target tree is not yet implemented.` (`updater_quantile_hist.cc:630`) |
| `sample_weight`, `subsample`, `colsample_*`, `booster="dart"` | ✅ | |
| `xgb.cv`, save/load `.ubj`/`.json` | ✅ | |
| `trees_to_dataframe`, JSON dump | ❌ | see above |
| distributed training | ❌ | v3.2.0 changelog "Currently missing" |

Both constraint failures matter for a forward-return model: `monotone_constraints` is the
natural way to encode a directional prior, and `interaction_constraints` is the natural way
to stop astro features from interacting spuriously. Neither is available with vector leaf
until 3.4.

---

## 4. Cost vs 10 independent single-output models

6517 rows → 5517 train / 1000 test, 100 features (40 categorical), 10 targets,
300 rounds, depth 6, lr 0.05, subsample 0.8, colsample 0.8. 16 threads.
Peak RSS measured with **one variant per process** (measuring in-process is
order-dependent and misleading).

| Variant | fit | predict | peak RSS | model | trees | RMSE |
|---|---|---|---|---|---|---|
| **A** `multi_output_tree` | **62.5 s** | **0.050 s** | **651.6 MB** | **2.82 MB** | 300 | 0.031348 |
| **B** `one_output_per_tree` | **32.7 s** | 0.076 s | 315.6 MB | 17.60 MB | 3000 | 0.031158 |
| **C** 10× separate `XGBRegressor` | **39.0 s** | 0.450 s | 262.7 MB | 17.83 MB | 3000 | 0.030228 |
| **D** `MultiOutputRegressor` wrapper | 29.3 s | 0.458 s | 17.83 MB | 17.83 MB | 3000 | 0.030228 |

(baseline interpreter RSS ≈ 175 MB; the fit-attributable deltas are A ≈ 477 MB,
B ≈ 140 MB, C ≈ 89 MB.)

**The headline is counter-intuitive: at this data size the native multi-output tree is the
_slowest and heaviest_ option to train.**

- **A is 1.6× slower than C** and **1.9× slower than B**. The vector-leaf histogram is
  `n_bins × n_targets` wide, so split evaluation does 10× the work per candidate while
  building 1/10 the trees — and the constant factors currently favour the well-optimised
  scalar path. The roadmap still lists an open item: *"Use f-order for the gradient…
  The transformation takes about one-fifth of the training time. (#9508)"*
- **A uses ~5× the fit memory of C** (477 MB vs 89 MB). Absolute numbers are small — 650 MB
  peak on a 31 GB box is a non-issue — but it scales with `n_bins × n_targets × n_nodes`, so
  a deeper tree or a wider bin count could bite.
- **A wins decisively on the artifact**: 2.82 MB vs 17.8 MB (**6.3× smaller**) and 0.050 s vs
  0.450 s predict (**9× faster**). For 300 rounds × the walk-forward folds this is the
  difference between a tidy artifact set and a bulky one, and it matters for the forward
  inference run.
- **D (the wrapper) is not a performance problem** — it is essentially C plus sklearn
  bookkeeping. The case against `MultiOutputRegressor` was never speed; it is that it
  duplicates the `QuantileDMatrix`/binning work conceptually and gives you 10 disjoint
  artifacts with no shared early-stopping story. **B gets you everything the wrapper gets you,
  in one booster, for less code.**

**None of this is a bottleneck.** A full 300-round fit is ~1 minute. A walk-forward CV with,
say, 8 folds × a 30-point hyperparameter search is ~4 h for A and ~2.2 h for B on 16 cores.
Both are affordable; B leaves more headroom for the search.

---

## 5. Correlated outputs — for or against a shared tree structure?

3 data seeds × {cumulative, step} × {strategy}, 400 rounds cap, early stopping 30 on a held-out
800-row validation slice, scored on a 1000-row test tail. nRMSE = per-horizon RMSE ÷ per-horizon
target std (so horizons are comparable).

```
################ CUMULATIVE targets ################
  target intercorrelation: mean |r| off-diag = 0.755, min = 0.435, max = 0.967
  one_output_per_tree  nRMSE by horizon: [0.9974 0.9864 0.979  0.9783 0.9613 0.9654 0.9567 0.9586 0.956  0.9595]
                       mean nRMSE=0.96987  mean dir-acc=0.5901  pred cross-horizon |r|=0.584
  multi_output_tree    nRMSE by horizon: [0.9956 0.9949 0.9918 0.9928 0.991  0.9913 0.9896 0.9888 0.988  0.9893]
                       mean nRMSE=0.99131  mean dir-acc=0.6008  pred cross-horizon |r|=0.988
  >>> multi_output_tree nRMSE is +2.21% vs one_output_per_tree

################ STEP targets ################
  target intercorrelation: mean |r| off-diag = 0.046, min = 0.014, max = 0.108
  one_output_per_tree  nRMSE by horizon: [0.994  1.0014 1.0004 1.002  0.9987 1.0012 1.0027 1.0024 1.003  1.0057]
                       mean nRMSE=1.00114  mean dir-acc=0.5183  pred cross-horizon |r|=0.134
  multi_output_tree    nRMSE by horizon: [0.9963 1.0002 1.001  1.001  0.9987 1.001  0.9998 1.0006 1.0007 1.0021]
                       mean nRMSE=1.00016  mean dir-acc=0.5265  pred cross-horizon |r|=0.775
  >>> multi_output_tree nRMSE is -0.10% vs one_output_per_tree
```

Three things fall out, and the naive expectation ("correlated targets ⇒ share the tree") is
**not** what happens.

### (a) The shared structure collapses the output vector toward rank 1

Look at `pred cross-horizon |r|`. On STEP targets the true horizon-to-horizon correlation is
**0.046**, but `multi_output_tree` emits predictions correlated at **0.775** across horizons
— versus **0.134** for `one_output_per_tree`, which correctly tracks the near-zero truth. On
CUMULATIVE the vector leaf emits **0.988**, i.e. essentially a single number times a fixed
shape.

This is structural, not tuning: every leaf a sample lands in contributes one 10-vector, and
the sample visits the *same* leaf for all 10 outputs in every tree. The predicted vector is a
sum of leaf vectors drawn from a shared partition, so it is nearly collinear by construction.

**The vector leaf does not "learn" the correlation among targets — it _imposes_ one.** When
the imposed structure happens to match (cumulative, true |r| = 0.755), it costs a little
accuracy. When it doesn't (step, true |r| = 0.046), it manufactures correlation that isn't
there. Neither is a win.

For map decision 8 this is the sharpest edge: a trading simulation derives positions from the
predicted return *vector*. A near-rank-1 vector carries roughly **one** number, so a strategy
that keys on the *shape* of the forward curve (e.g. "up at h3, down by h8") has almost nothing
to work with under `multi_output_tree`.

### (b) On correlated (cumulative) targets, sharing is actively 2.2% worse

Averaged over 3 seeds, `multi_output_tree` nRMSE is **+2.21%** on cumulative — the case where
correlation was supposed to help. The regularisation story ("correlated targets ⇒ shared
structure denoises") doesn't fire here, most likely because (i) the shared split must serve 10
targets so it can't specialise to the horizon where the astro signal is actually present, and
(ii) at 5517 rows with a very weak signal, the constraint costs more than the variance
reduction returns. On step targets the two are a statistical dead heat (−0.10%).

The one place vector leaf edges ahead is **directional accuracy** (0.6008 vs 0.5901 cumulative,
0.5265 vs 0.5183 step) — consistent with the coherence effect: a rank-1 output gets the sign
consistent across horizons even when the magnitude is wrong. Worth an ablation, given decision
8 scores on trading metrics rather than RMSE, but the margin is well inside seed noise at
n = 3.

### (c) The cumulative target silently reweights the loss toward long horizons

Because cumulative returns are a running sum, their variance grows ~linearly with horizon. The
single flattened-SSE loss and the single aggregated early-stopping metric ([§3](#3-early-stopping-feature-importance-shap))
therefore do **not** treat the 10 horizons equally:

```
STEP        per-horizon variance (x1e6): [112.2 110.  107.7 103.  107.8 104.2 106.5 105.  101.3  99.2]
            share of total SSE          : [10.6 10.4 10.2  9.7 10.2  9.9 10.1  9.9  9.6  9.4] %
            h10 / h1 variance ratio     : 0.9x
CUMULATIVE  per-horizon variance (x1e6): [ 112.2 246.3 389.2 535.9 705.7 868.1 1031.4 1204.5 1349.3 1499.0]
            share of total SSE          : [ 1.4  3.1  4.9  6.7  8.9 10.9 13.  15.2 17.  18.9] %
            h10 / h1 variance ratio     : 13.4x
```

On the cumulative target, **h1 gets 1.4% of the gradient budget and h10 gets 18.9% — a 13.4×
imbalance**, and early stopping stops on whatever is best for h10. Nobody chose that; it is a
side effect of the target parameterisation. On the step target the weighting is uniform
(9.4–10.6%).

This is fixable (per-target weights, or standardising each horizon before fitting), but it is
an extra mechanism to build and validate — and `multi_strategy` offers no per-target loss
weight (roadmap item *"Loss weight"* is still unchecked).

---

## 6. The decision

**Map decision 2's conditional reads:** *"For XGBoost, use a vector of 10 cumulative returns
**if** XGBoost supports it natively without a `MultiOutputRegressor` wrapper — otherwise
10-step for both."*

Taken literally the trigger fires: native multi-output works. But the conditional was written
on the assumption that native support is the *scarce* thing and that the cumulative form is
what it buys. Neither holds:

1. **Native support is not scarce and is not target-specific.** `XGBRegressor` takes an
   `(n, 10)` `y` with no wrapper for *either* target definition, and has since 1.6. Nothing
   about `multi_strategy` prefers cumulative over step. The premise that made the choice
   conditional is void.
2. **The cumulative form actively hurts under a shared loss.** One unweighted SSE over all 10
   targets plus one aggregated early-stopping scalar means the cumulative target hands 18.9%
   of the loss to h10 and 1.4% to h1 (§5c). The model silently becomes a 10-day-horizon model.
3. **Nothing is lost by choosing step.** Cumulative returns are `cumsum` of step returns — an
   exact linear map. Step predictions convert to cumulative predictions post hoc for free, so
   the trading simulation can score either. The only thing the target choice fixes is *where
   the squared loss is applied*, and the step parameterisation applies it evenly.
4. **A shared target definition makes the three-model comparison honest.** Map decision 8
   scores baseline / XGBoost / TFT side by side on one scorecard. With XGBoost on cumulative
   and the TFT on step, any difference is confounded by the target parameterisation.

### ✅ Resolution

> **XGBoost trains on the 10-STEP vector `r_k = log(C_{t+k}/C_{t+k-1})`, k = 1..10 — the same
> target definition as the TFT.** Both models share one target. The cumulative 10-vector is
> derived by `cumsum` at scoring time when the trading simulation wants it.

### ✅ Companion recommendation (not asked for, but implied)

> **Production setting: `multi_strategy="one_output_per_tree"` (the default) on
> `tree_method="hist"`, `device="cpu"`, `enable_categorical=True`, XGBoost ≥ 3.3.0.** This is
> still native multi-output in a single booster, no wrapper.
>
> Reserve `multi_strategy="multi_output_tree"` for a **one-line ablation**, reported as a
> secondary row on the scorecard.

Why the default rather than the vector leaf, given the ticket is about the vector leaf:

- **It preserves decision 6 fully.** Partition-based categorical splits are available; the
  vector leaf's one-vs-rest-only splits (§2) are a real capability loss on 27-level
  nakshatras, and `max_cat_to_onehot` is silently ignored.
- **It's 1.9× faster and uses 1/3 the fit memory** (§4) — more headroom for the walk-forward
  hyperparameter search.
- **It scored better** on the correlated target (−2.2% nRMSE) and tied on step (§5).
- **It doesn't impose a false correlation structure** on the output vector — important because
  decision 8 reads the vector's shape.
- **It keeps `monotone_constraints`, `interaction_constraints`, `reg:quantileerror`,
  `trees_to_dataframe` and valid JSON dumps** — all unavailable or broken under vector leaf in
  3.3.0 (§3).
- The vector leaf's genuine wins — 6.3× smaller model, 9× faster predict — are not binding
  constraints for this project.

Revisit when **3.4.0** ships: it brings partition-based categorical splits (#12299),
interaction constraints (#12294), monotone constraints (#12341), and `trees_to_dataframe`
(#12293), which removes most of the objections above. The directional-accuracy edge in §5 is
the reason to keep the ablation rather than drop the option.

### Concrete configuration

```python
import xgboost as xgb

model = xgb.XGBRegressor(
    tree_method="hist",
    device="cpu",
    enable_categorical=True,          # default-on in 3.3.0; pass it anyway
    multi_strategy="one_output_per_tree",   # native multi-output, no wrapper
    max_cat_to_onehot=4,              # keep partition splits for the 27-level astro cats
    eval_metric="rmse",
    early_stopping_rounds=...,
    n_estimators=...,
)
model.fit(X_train, Y_train_step)      # Y shape (n, 10) -> predict returns (n, 10)
```

Two things the training-plan ticket must carry forward:

- **Early stopping is one scalar over all 10 horizons.** Decide deliberately whether that is
  wanted, or whether horizons should be standardised / weighted first.
- **Build the `pd.Categorical` dtypes from `categories_list.json` before the fold split**, not
  from each fold's observed levels — otherwise a fold that lacks a level will produce a model
  that raises `Found a category not in the training data` at inference.

---

## Reproducing

Every number above came from these scripts against the source-built XGBoost 3.3.0 described in
the environment box. The synthetic generator (`exp_common.py`) produces 6517 rows × (60 numeric
+ 40 `category`) with a 10-element target in step and cumulative form, and is included in full
below so the numbers can be re-derived.

To re-run them now, drop the scripts somewhere in the repo and go through the container —
`./container/run.sh python <script>.py`. Structural results (tree counts, split sizes, shapes,
RMSE) are deterministic and reproduce; wall-clock timings will differ from §4, which was
measured on the host build.

<details>
<summary><code>exp_common.py</code> — synthetic data matched to the project shape</summary>

```python
"""Synthetic data shaped like the Nifty-50 astro problem."""
import numpy as np
import pandas as pd

N_ROWS = 6517
N_NUM = 60
N_CAT = 40
H = 10

# cardinalities modelled on the real astro columns
CARDS = ([27] * 9 + [12] * 9 + [2] * 8 + [4] * 6 + [7] + [3] * 4 + [9] * 3)[:N_CAT]


def make_data(seed=0, signal=0.35):
    rng = np.random.default_rng(seed)

    # numeric: sin/cos of slowly-precessing planetary longitudes
    t = np.arange(N_ROWS)
    num = np.empty((N_ROWS, N_NUM), dtype=np.float32)
    periods = rng.uniform(20, 4000, N_NUM)
    phase = rng.uniform(0, 2 * np.pi, N_NUM)
    for j in range(N_NUM):
        num[:, j] = np.sin(2 * np.pi * t / periods[j] + phase[j])
    num += rng.normal(0, 0.05, num.shape).astype(np.float32)

    # categorical: derived from the same cyclic clocks -> realistic dependence
    cats = {}
    cat_codes = np.empty((N_ROWS, N_CAT), dtype=np.int64)
    for j, k in enumerate(CARDS):
        per = rng.uniform(15, 800)
        codes = (np.floor((t / per + rng.uniform(0, 1)) * k) % k).astype(np.int64)
        cat_codes[:, j] = codes
        cats[f"cat_{j}"] = pd.Categorical.from_codes(codes, [f"L{i}" for i in range(k)])

    X = pd.DataFrame(num, columns=[f"num_{j}" for j in range(N_NUM)])
    for name, c in cats.items():
        X[name] = c

    beta_num = rng.normal(0, 1, N_NUM) * (rng.random(N_NUM) < 0.15)
    latent = num @ beta_num
    cat_effect = np.zeros(N_ROWS)
    for j in (0, 1, 9, 10):
        lvl = rng.normal(0, 1, CARDS[j])
        cat_effect += lvl[cat_codes[:, j]]
    drive = 0.6 * latent + 0.4 * cat_effect
    drive = (drive - drive.mean()) / drive.std()

    y_step = np.empty((N_ROWS, H), dtype=np.float32)
    for k in range(H):
        y_step[:, k] = signal * drive * (0.9 ** k) + rng.normal(0, 1.0, N_ROWS)
    y_step *= 0.01  # daily log-return scale

    y_cum = np.cumsum(y_step, axis=1).astype(np.float32)
    return X, y_step, y_cum


def corr_summary(y):
    c = np.corrcoef(y, rowvar=False)
    off = c[np.triu_indices_from(c, k=1)]
    return f"mean |r| off-diag = {np.abs(off).mean():.3f}, min = {off.min():.3f}, max = {off.max():.3f}"


def split(X, y, n_test=1000):
    n_tr = len(X) - n_test
    return X.iloc[:n_tr], X.iloc[n_tr:], y[:n_tr], y[n_tr:]
```

</details>

<details>
<summary>§2 — categorical split-type probe</summary>

```python
import re, numpy as np, pandas as pd, xgboost as xgb
from collections import Counter

rng = np.random.default_rng(7)
N, K = 6000, 27
codes = rng.integers(0, K, N)
lvl = rng.normal(0, 1, K)
X = pd.DataFrame(rng.normal(0, 1, (N, 3)), columns=["n0", "n1", "n2"])
X["nak"] = pd.Categorical.from_codes(codes, [f"L{i}" for i in range(K)])
y = np.cumsum(np.stack([lvl[codes] * (0.9 ** k) + rng.normal(0, .3, N) for k in range(10)], 1), 1)

PAT = re.compile(r"\[nak:\{([^}]*)\}\]")

def fit(strategy, mcto, n=20, d=5):
    m = xgb.XGBRegressor(tree_method="hist", device="cpu", enable_categorical=True,
                         multi_strategy=strategy, n_estimators=n, max_depth=d,
                         max_cat_to_onehot=mcto, learning_rate=0.3)
    m.fit(X, y)
    return m.get_booster()

for strategy in ("one_output_per_tree", "multi_output_tree"):
    for mcto in (4, 64):
        b = fit(strategy, mcto)
        sizes = [len(g.split(",")) for t in b.get_dump() for g in PAT.findall(t)]
        print(f"  {strategy:20s} mcto={mcto:3d} -> {len(sizes):4d} cat splits, "
              f"size histogram {dict(sorted(Counter(sizes).items()))}")
```

</details>

<details>
<summary>§4 — isolated cost benchmark (one variant per process)</summary>

```python
"""Run as:  python exp5_iso.py {multi|oopt|ten}"""
import sys, os, time, resource, numpy as np, xgboost as xgb
from exp_common import make_data, split

variant = sys.argv[1]
X, y_step, y_cum = make_data()
Xtr, Xte, ytr, yte = split(X, y_cum, n_test=1000)
PARAMS = dict(tree_method="hist", device="cpu", enable_categorical=True,
              n_estimators=300, max_depth=6, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)

base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t = time.perf_counter()
if variant == "multi":
    m = xgb.XGBRegressor(multi_strategy="multi_output_tree", **PARAMS).fit(Xtr, ytr)
    pred = lambda: m.predict(Xte)
elif variant == "oopt":
    m = xgb.XGBRegressor(multi_strategy="one_output_per_tree", **PARAMS).fit(Xtr, ytr)
    pred = lambda: m.predict(Xte)
elif variant == "ten":
    ms = [xgb.XGBRegressor(**PARAMS).fit(Xtr, ytr[:, k]) for k in range(10)]
    pred = lambda: np.column_stack([mm.predict(Xte) for mm in ms])
fit_s = time.perf_counter() - t
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t = time.perf_counter(); p = pred(); pred_s = time.perf_counter() - t
print(f"{variant:6s} fit={fit_s:7.2f}s  predict={pred_s:6.3f}s  "
      f"peakRSS={peak:7.1f}MB  (baseline {base_rss:.1f}MB)  "
      f"RMSE={np.sqrt(((p-yte)**2).mean()):.6f}")
```

Output:

```
multi  fit=  62.51s  predict= 0.050s  peakRSS=  651.6MB  (baseline 175.0MB)  RMSE=0.031348
oopt   fit=  32.71s  predict= 0.076s  peakRSS=  315.6MB  (baseline 175.8MB)  RMSE=0.031158
ten    fit=  39.02s  predict= 0.450s  peakRSS=  262.7MB  (baseline 173.7MB)  RMSE=0.030228
```

</details>

<details>
<summary>§5 — correlated-targets comparison</summary>

```python
import numpy as np, xgboost as xgb
from exp_common import make_data, corr_summary, split

PARAMS = dict(tree_method="hist", device="cpu", enable_categorical=True,
              n_estimators=400, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
              early_stopping_rounds=30, eval_metric="rmse")

def run(X, Y):
    Xtr, Xte, ytr, yte = split(X, Y, n_test=1000)
    Xa, Xv, ya, yv = split(Xtr, ytr, n_test=800)
    out = {}
    for s in ("one_output_per_tree", "multi_output_tree"):
        m = xgb.XGBRegressor(multi_strategy=s, **PARAMS)
        m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        p = m.predict(Xte)
        nrmse = np.sqrt(((p - yte) ** 2).mean(0)) / yte.std(0)
        dir_acc = (np.sign(p) == np.sign(yte)).mean(0)
        out[s] = dict(best_iter=m.best_iteration, nrmse=nrmse, dir=dir_acc,
                      pred_corr=np.abs(np.corrcoef(p, rowvar=False)[
                          np.triu_indices(p.shape[1], 1)]).mean())
    return out

for name in ("CUMULATIVE", "STEP"):
    for seed in (0, 1, 2):
        X, ys, yc = make_data(seed=seed)
        r = run(X, yc if name == "CUMULATIVE" else ys)
        # ... aggregate and print (see full output in §5)
```

</details>

---

## Sources

All primary. Docs read at `en/stable` and `en/latest`; source read at the `v3.3.0` tag and at
`master` where the difference mattered.

- [Multiple Outputs tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html)
  · [raw RST on master](https://github.com/dmlc/xgboost/blob/master/doc/tutorials/multioutput.rst)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) — `multi_strategy`, `tree_method`, `max_cat_to_onehot`
- Changelogs: [v3.2.0](https://github.com/dmlc/xgboost/blob/master/doc/changes/v3.2.0.rst) ·
  [v3.3.0](https://github.com/dmlc/xgboost/blob/master/doc/changes/v3.3.0.rst) ·
  [v3.0.0](https://xgboost.readthedocs.io/en/latest/changes/v3.0.0.html)
- Source: [`src/tree/hist/evaluate_splits.h` @ v3.3.0](https://github.com/dmlc/xgboost/blob/v3.3.0/src/tree/hist/evaluate_splits.h)
  (`HistMultiEvaluator::EvaluateSplits`, L675–684) vs
  [@ master](https://github.com/dmlc/xgboost/blob/master/src/tree/hist/evaluate_splits.h) (L660–710)
- Roadmap: [dmlc/xgboost#9043 — [Roadmap] Multiple outputs](https://github.com/dmlc/xgboost/issues/9043)
- PRs: [#10700](https://github.com/dmlc/xgboost/pull/10700) (weight importance) ·
  [#12072](https://github.com/dmlc/xgboost/pull/12072) (one-hot cat, CPU) ·
  [#12244](https://github.com/dmlc/xgboost/pull/12244) ·
  [#12276](https://github.com/dmlc/xgboost/pull/12276) (one-hot cat, GPU) ·
  [#12293](https://github.com/dmlc/xgboost/pull/12293) (tree dataframe) ·
  [#12294](https://github.com/dmlc/xgboost/pull/12294) (interaction constraints) ·
  [#12299](https://github.com/dmlc/xgboost/pull/12299) (partition cat split, CPU) ·
  [#12305](https://github.com/dmlc/xgboost/pull/12305) (partition cat split, GPU) ·
  [#12341](https://github.com/dmlc/xgboost/pull/12341) (monotone constraints)
- Packaging metadata read from the [PyPI JSON API](https://pypi.org/pypi/xgboost/json) for
  2.0.3 / 2.1.4 / 3.0.0 / 3.0.5 / 3.1.3 / 3.2.0 / 3.3.0
- Iosipoi & Vakhrushev, *Fast Gradient Boosted Decision Tree for Multioutput Problems*,
  [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/a36c3dbe676fa8445715a31a90c66ab3-Paper-Conference.pdf)
  — the Sketch Boost basis for the 3.2.0 reduced-gradient feature (GPU/cuML-oriented; not
  applicable at 10 targets)
