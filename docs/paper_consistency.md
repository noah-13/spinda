# SPINDA paper-to-repository consistency record

Checked against the local manuscript *SPINDA: Simple Prediction and
Interpretation of Data with Human Label Variation*.

| Paper capability | Public implementation |
| --- | --- |
| Soft CE, soft MSE, soft JSD, ReL, and hard-CE training | `hlv_toolkits.scripts.train` and `models.trainer` |
| Single-label, multi-label, and multi-dimensional data | `hlv_toolkits.data.readers` |
| Model-independent prediction contract | [prediction contract](prediction_contract.md) and `evaluate` |
| Distribution-aware metrics and visualisation | `hlv_toolkits.eval` and `visualization` |
| Entropy-stratified and instance-level analysis | `evaluate --analysis` plus seed-aggregated `spinda analyze` plots |
| Six paper dataset families | [reproduction launchers](reproducing_paper.md) |

## Release decisions

- **SPINDA** is the public project name; `hlv_toolkits` remains the import
  namespace for backwards compatibility.
- Generated runs, analyses, dashboards, manuscript copies, and editor backups
  are ignored; they are reproducible artifacts rather than source release files.
- Only the launcher table in `reproducing_paper.md` is the supported public
  reproduction surface.

## Metadata still needed

The local manuscript is anonymous and has placeholder links. Before announcing
the GitHub repository, add the final author list, ACL Anthology/DOI and video
URLs. These cannot safely be inferred from
an anonymous submission.
