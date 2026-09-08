# UFNet paper-aligned manifest audit

Status: **PASS**

Official source commit: `5ece2c65ba184faccf6c8cdccdc03132427c464b`

## Paper count gate

| split | participants_observed | participants_expected | sessions_observed | sessions_expected | participants_match | sessions_match |
| --- | --- | --- | --- | --- | --- | --- |
| train | 516 | 516 | 690 | 690 | True | True |
| validation | 167 | 167 | 215 | 215 | True | True |
| test | 162 | 162 | 197 | 197 | True | True |
| all | 845 | 845 | 1102 | 1102 | True | True |

## Interpretation

- `all_task_ids.txt` defines the 845-participant three-task fusion cohort.
- The released validation/test lists each contain 267 participants because they also cover the larger unimodal cohorts.
- Intersecting those lists with the fusion cohort yields 167 validation and 162 test participants; the remaining 516 are training participants.
- Sessions are reconstructed with the exact released-code key: parsed `participant_id#YYYY-MM-DD`; finger tapping first requires both left and right hands, then finger/speech/smile are inner-joined.
- The released join key is not unique when a participant repeats tasks on the same day. The manifest preserves the paper code's Cartesian combinations and adds a deterministic `manifest_row_id`; deduplicating the date key would incorrectly reduce 1,102 sessions to 1,013.
- No row was deleted merely to force a paper count. Any failed check makes this manifest non-authoritative.

## Integrity checks

```json
{
  "exact_paper_counts": true,
  "participant_lists_unique": true,
  "dev_test_disjoint_within_fusion_cohort": true,
  "all_session_ids_consistent": true,
  "all_session_labels_consistent": true,
  "all_sessions_in_all_task_list": true,
  "all_listed_participants_have_sessions": true,
  "unique_manifest_row_ids": true,
  "no_duplicate_source_combinations": true
}
```

## Files

- `paper_session_manifest.csv`: immutable session-level split and source-file mapping.
- `paper_participant_manifest.csv`: immutable participant-level split, labels, demographics, and session counts.
- `paper_partition_summary.csv`: expected-versus-observed gate.
- `paper_modality_presence_and_exclusions.csv`: modality availability and explicit exclusion reason.
- `source_provenance.json`: source hashes, commit, raw list sizes, and checks.
