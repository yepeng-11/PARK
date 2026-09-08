# Why the prior server dataset did not reproduce the UFNet paper cohort

## Conclusion

The prior experiment used the later `ROC-HCI/PARK` repository and its cross-setting validation data, not the released `ROC-HCI/UFNet` AAAI 2025 repository and split contract. The discrepancy is therefore a protocol/data-version mismatch, not evidence that the paper's 845-person cohort is absent from the released data.

## Evidence captured on the server

| Item | Prior server experiment | Paper-aligned source |
| --- | ---: | ---: |
| Repository | `ROC-HCI/PARK` | `ROC-HCI/UFNet` |
| Commit | `c129e9f16c24ebba7248ebef5bc34666cbeade92` | `5ece2c65ba184faccf6c8cdccdc03132427c464b` |
| Finger feature table | 3,315 × 134 | 3,177 × 133 |
| Smile feature table | 1,773 × 54 | 1,684 × 54 |
| Speech feature table | 1,911 × 2,054 | 1,821 × 1,030 |
| Fusion table | 1,119 rows / 909 participants | 1,102 rows / 845 participants |
| Validation ID list | 120 IDs | 267 general IDs; 167 after intersection with `all_task_ids.txt` |
| Test ID list | 391 unique IDs | 267 general IDs; 162 after intersection with `all_task_ids.txt` |

All 845 official fusion participants occur in the PARK fusion table, but PARK adds 64 participants. Among the official 845, PARK has 1,055 fusion rows rather than the official-code result of 1,102. Thus it is neither the same cohort nor the same session construction.

The PARK split files are also not renamed copies of the paper split: within the official 845-person cohort, the PARK dev/test lists select 70/212 people. Only 13 of the paper's 167 validation participants and 40 of its 162 test participants retain the same role. This is a different experimental partition.

## Two subtle implementation traps

1. The official dev and test files each contain 267 participants because they serve both unimodal and fusion experiments. For three-task fusion, they must be intersected with the 845 IDs in `all_task_ids.txt`; this produces 167 validation, 162 test, and 516 training participants.
2. The released fusion code joins on `participant_id#YYYY-MM-DD`. This key is non-unique for repeat recordings. Its joins yield 1,102 rows from 1,013 unique date keys. Deduplicating by the date key removes 89 rows and is not paper-aligned.

## Frozen decision

Use the manifests in this directory for the official-split reproduction only. Do not regenerate official membership from PARK's feature tables or its validation lists. Any later feature replacement must join to `manifest_row_id`/the four recorded source filenames and must pass the participant and session count gate before training.

The nested participant-disjoint five-fold protocol remains the stronger main experiment; this manifest is specifically for reproducing and comparing with the paper's official split.
