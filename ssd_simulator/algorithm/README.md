# RANCHO for SSD

This is the SSD policy used for the recorded trace results. It estimates how much prompt computation a write can save, then decides whether to spend the write budget on it.

1. **Build training examples.** A block leaving GPU memory is one example. The label counts how much extra prompt input would have been readable if that block alone had been available, using a fixed reference replay.
2. **Train at each hour.** The model uses four features from the last request: user ID, turn number, chat age and time since the previous request. It also uses four features about the block: past accesses, past evictions, time in memory and time since its last access. The fit only uses reuse observed before that hour.
3. **Decide whether to write.** The simulator scores offers, gives extra weight to blocks evicted repeatedly and checks what the budget can afford. It can also use a limited amount of spare hourly write budget.
4. **Count saved computation.** We count eligible reads within the same conversation that SSD serves beyond the GPU cache's readable prefix.

| File | Where to look |
| --- | --- |
| [run.py](run.py) | The steps from saved inputs to final counts |
| [training_data.py](training_data.py) | The eight inputs and which examples are visible at each hour |
| [model_training.py](model_training.py) | Tree fitting, validation and prediction |
| [recorded_settings.json](recorded_settings.json) | Recorded settings whose loss override is applied in [experiment_settings.py](experiment_settings.py) |
| [replay.cpp](replay.cpp) | Cache updates, cutoff refreshes and write decisions |
| [reuse_labels.py](reuse_labels.py) | The count of observed gains available at each hour |
| [recomputation_counts.py](recomputation_counts.py) | The final count of reads SSD saved |
| [support](support) | Tools that make the reuse index and label data |

The [verification record](SOURCE_VERIFICATION.json) documents the original reconstruction. The public copy uses clearer names and the same replay rule. See [run commands and required files](RUNNING.md).
