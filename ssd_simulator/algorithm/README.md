# Recorded SSD algorithm

This is the SSD admission algorithm reconstructed from the implementation used for the recorded-trace results. It includes the predictor, write-budget rules and the calculation of avoided KV recomputation.

## Follow the steps

1. **Build examples.** When a block leaves GPU memory, record the request and block history. For a fixed reference cache, count how much later input would become readable if that block were saved. The label-building code records those gains by the time they occurred.
2. **Train each hour.** Use only gains already observed at that hour. Four inputs describe the last request: user ID, turn number, conversation age and the gap between requests. Four more describe the block: access count, eviction count, time in GPU memory and time since its last access.
3. **Decide what to save.** Predict the benefit of saving an evicted block. Apply the recorded score adjustments and write-budget rules, then write accepted blocks to SSD. The rules include stronger scores for repeatedly evicted blocks and limited use of spare hourly write capacity.
4. **Count saved computation.** Find which later reads SSD served after the GPU cache lookup. Count only eligible repeated reads within the same conversation. This gives the reported recomputation percentage, not a measurement of GPU execution time.

## Code map

| File | Purpose |
| --- | --- |
| [run.py](run.py) | Train the hourly models, run the replay, check it and calculate the result. |
| [training_data.py](training_data.py) | Load checked arrays, build the eight inputs and select observations before each cutoff. |
| [training.py](training.py) | Fit the tree and predict reuse. The six-hour validation step chooses the number of trees. |
| [recorded_settings.json](recorded_settings.json) | Original recorded model settings. The runner applies the recorded raw-label loss override. |
| [replay.cpp](replay.cpp) | Execute the recorded SSD admission and write-budget policy. |
| [accounting.py](accounting.py) | Apply the same-conversation reuse index to the replay's request counts. |
| [labels.py](labels.py) | Accumulate reference-label gains strictly before each hourly cutoff. |
| [support/](support) | Native reference-label and session-accounting builders. These are supporting preparation tools. |

The smaller [baseline replay](../replay/README.md) remains available for comparisons and cache-correctness checks. The [SSD engine integration](../../ssd_serving/README.md) is a separate path for actual model execution and is still under development.

## What was verified

The reconstructed policy source, original worker and model recipe match the file fingerprints saved with the accepted SSD grid. The policy was then reformatted and its internal names clarified without changing its operations.

The readable policy also matched the reconstructed original on three synthetic 96-hour cases, including every write decision and per-request count. These cases exercise hourly score updates and spare-budget writes. The tree-only trainer produced identical native trees and predictions to the reconstructed original fitting code on two synthetic fitting cutoffs. An additional generated-data check ran all 96 fits, cache replay, independent checking and final reuse accounting. [Verification record](SOURCE_VERIFICATION.json).

**The full private-trace replay has not been rerun locally.** Matching source and synthetic behavior is different from reproducing the numerical trace results. The runner can check supplied prediction fingerprints and expected result counts when the authorized inputs are available. Private traces, labels, production predictions and models are not distributed here.

For commands and input formats, see [RUNNING.md](RUNNING.md).
