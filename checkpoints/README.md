# Checkpoints

Trained runs. `*.pt` is gitignored; `config.json` / `results.json` may be present.

## Keep (cited in analysis / blog)

| ID | Why |
|----|-----|
| `ns_direct_a3`, `a3_nu0p1_T5` | Classical TGV works |
| `ns_par_c_s0`, `ns_par_q_s0` | Best TGV hypernet comparison |
| `ns_par_c_v2_s0`, `ns_par_q_v2_s0` | expect / log-ν redesign |
| `kol_direct_s0`, `kol_par_c_s0`, `kol_par_q_s0` | Kolmogorov null |
| `burg_vqc_c_scout`, `burg_vqc_q_scout` | Hard-IC Burgers (valid null) |

## Historical / invalid (do not plot as success)

| Pattern | Why |
|---------|-----|
| `ns_run_0001`–`0005`, `ns_sweep_*` | Wrong pressure sign / collapse |
| `ns_direct_a3_time` | Proved dead time channel (duplicate of a3) |
| `burg_vqc_*_s0` without `hard_ic` / collapse diagnostics | Soft-IC collapse |
| `ns_direct_smoke_*`, `smoke_kol` | Smoke tests |

New demo models should use new run-ids (`tgv_demo_*`), not overwrite these.
