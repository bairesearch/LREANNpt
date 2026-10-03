Reproducing the full-data population perturbation experiment
==========================================================

From the repository root, on Linux x86-64 with an NVIDIA GPU:

    conda create -n suann-benchmark-repro python=3.12 pip
    conda activate suann-benchmark-repro
    cd population_pertubation_train_convergence_full
    python -m pip install -r requirements.txt
    python reproduce.py

By default, the output directory is the folder containing reproduce.py, even
when the command is invoked from another working directory. A fresh launch
overwrites saved experiment results in that folder. This single launch command
records the code/input snapshot, downloads the pinned sources, rebuilds missing
full-sized splits, checks their hashes against the reference manifests, performs
numerical verification, trains all 195 jobs, independently checks final metrics, and writes
REPORT.txt, summary/per-seed CSV files, and PNG/SVG figures in that output folder.
No production configuration edits or manual setsid command are required.
The supplied shell wrapper accepts the same arguments, for example:

    bash population_pertubation_train_convergence_full.sh

Use either the Python command or the shell wrapper for a given launch.
The command stays in the foreground; manager output goes to logs/manager.log.
A compatible NVIDIA driver for the pinned PyTorch CUDA 12.8 build is required.

The three-worker GPU schedule, populations 64/256/1024/4096, Adam baseline, seeds
11/22/33, and benchmark preprocessing are unchanged. Checkpoint selection and
stopping use validation loss by default, or the original training-loss policy. Banking Marketing
now uses bank-full.csv with disjoint splits, as described below; the other
dataset partitions are unchanged.
There are no source, training, validation, or test row caps, and no maximum
training-iteration or wall-time cutoff. Plan for tens of GB of disk space, plus
model checkpoints, and substantial GPU time. The larger dataset preparation
steps also need several GB of RAM. Training uses the frozen source/LREANNpt
modules, not the working production directory.

Training or validation loss selection and stopping
--------------------------------------------------

Set "trainSetLossOptimisation" in protocol.json before launching a fresh benchmark:

    false (default): select minimum full-validation cross-entropy and stop when
                    validation loss stabilises. Accuracy does not select models
                    or reset patience; there is no perfect-training-fit stop.
    true:           restore the original full-training-loss policy, including
                    stopping at 100% training accuracy and training CE <= 0.01
                    once the minimum training period has elapsed.

The one option applies to Adam and all population sizes. Both modes retain the
benchmark's existing train/validation/test partitions. In training-loss mode,
validation metrics are diagnostic only. Equal selected-split losses retain the
earlier checkpoint regardless of accuracy. Parameter updates and population
rewards always use training minibatches only.

Evaluate every 100 updates, with the existing minimum of max(2,000 updates, one
complete configured training pass). A selected-split loss decrease greater than
max(0.0001, 0.001 * last significant best loss) resets patience. Smaller decreases
can improve the selected checkpoint without resetting patience. After 2,000 updates
without significant improvement, restore the best checkpoint and matching optimizer
state and multiply the learning rate by 0.2. After three reductions, the next
plateau stops training. Numerical recovery restores a finite selected checkpoint
and counts as a learning-rate reduction. There is no fixed maximum update/epoch/time
limit. Validation accuracy remains a diagnostic metric in validation-loss mode.

Final train, validation, and test metrics all describe the same restored selected
checkpoint. REPORT.txt includes separate accuracy tables (mean and sample standard
deviation over three seeds) for all three splits, every dataset, and every method,
plus a side-by-side table with train/validation CE and stopping updates. Incomplete
groups stay pending. All split metrics are also retained in the JSON and CSV files.
The test set is evaluated only after stopping and restoration in either mode.

REPORT.txt also includes an elapsed-time table with one row per dataset and
columns for Adam and populations 64/256/1024/4096, matching the accuracy tables.
Each cell shows elapsed run time as mean +/- sample standard deviation across
three seeds in H:MM:SS, rounded to the nearest second; incomplete groups show
"pending". These are per-seed averages, not sums across seeds. Elapsed run time
includes updates, evaluation and checkpointing through final metrics, but excludes
data preparation, initial setup, queue waits and downtime between resumes. It
includes work after the selected checkpoint until stopping. Exact seconds for
both update time and elapsed time (mean, SD and three-seed total) remain in
summary.json/summary.csv, with individual times in per_seed.csv. Update time
excludes full-split evaluation and checkpoint saves. Because workers run
concurrently, summed run durations are not the wall-clock duration of the experiment.

This is the selectable-loss-v1 protocol. It corrects the previous upgrade's
validation-accuracy objective to validation loss. The boolean and selection metric
are saved in each run's protocol and convergence policy. Start a fresh reproduction
to adopt this code or change modes; earlier checkpoints/results and different modes
cannot be resumed or pooled under this protocol. Existing separate output folders
retain their own frozen code, protocol, and reports. Updating this repository does
not modify a running copy. Historical reports and saved results are not regenerated.

In main LREANNpt, set trainSetLossOptimisation in LREANNpt_SUANN_globalDefs.py;
it is used only when populationPertubationOptimiseTrainingIterations=True. It
also defaults to False. False retains an existing tabular validation split or
holds out 20% of training rows before fitting category/normalisation statistics
or repeating training rows. CIFAR-10 validation is held out from training with
deterministic test transforms. The test split is never used as validation. The
split settings are populationPertubationValidationSplitSize and
populationPertubationValidationSplitSeed. True skips the added holdout and all
validation evaluations in the production training loop, preserving the original
training rows, preprocessing and random-number usage. Existing validation splits
are not merged into training. Ordinary production Adam training is unchanged;
benchmark Adam explicitly uses the shared convergence controller.

Fresh runs and resuming
----------------------

Without --resume, a launch in this folder clears runs/, checkpoints/, figures/,
execution logs, generated reports/CSV files, top-level result plots, runtime
statuses (including CANCEL), and per-run verification records. Training starts
from fresh models. Code, protocol, inputs, reference manifests, source downloads,
prepared arrays and software-validation evidence are retained. Input caches are
verified before reuse; previous training results are never reused in a fresh run.
The environment check and existing launcher lock precede the in-place reset.

To write a separate reproduction, the existing --output option is still available:

    python reproduce.py --output ../../population_pertubation_reproduction_001

A separate fresh output directory must not already exist. The launcher copies
code, small inputs and reference manifests, but never copies archived results,
arrays, checkpoints, runtime statuses, or old numerical verification gates.
All paths, including sources and working checkpoints, are local to the output
directory.
There is no dependency on /tmp/suann-population-train-convergence-full, and
separate reproductions cannot resume one another's checkpoints implicitly.

To explicitly resume the SAME reproduction after interruption:

    python reproduce.py --resume

For a separate output folder, include its --output path with --resume.
Resume preserves results and verifies the saved code/input/protocol snapshot
before continuing from that folder's own checkpoints. When resuming a separate
output folder, changes in the original folder are not copied in.
Do not edit a run's snapshot to change its scientific protocol; start a new run.
Resume on the same GPU/software environment for the closest numerical replay.
Changing GPU hardware can change floating-point results and training trajectories.

To prepare and audit data without starting training:

    python reproduce.py --prepare-only

This is also a fresh launch: it resets saved results in this folder. Add --resume
to recheck an already initialised run without resetting its saved training state,
or supply a new --output folder to keep the current folder's results.

For a quick check using COMPLETE small datasets (no sampling):

    python reproduce.py --output ../../population_data_check_small --prepare-only --datasets iris new-thyroid titanic

Later, use --resume without --prepare-only/--datasets to prepare any remaining
datasets and train the entire 195-run protocol. --datasets is only a preparation
option; it never silently reduces the training experiment's scope.

Re-running preparation checks every required array's size and SHA-256. Missing
or corrupt arrays are rebuilt, even when manifest.json still exists. Rebuilt
arrays, split indices, feature metadata, category maps and normalisation
statistics must match the archived references before a new manifest is accepted.
A difference is an error, not an automatic update of the expected experiment.

To stop the manager, press Ctrl-C in its launcher, or create OUTPUT/CANCEL.
The manager owns a separate process group, so cancellation targets that run.
If CANCEL exists, remove it explicitly before resuming. A completed archive
cleaned of model checkpoints is a record, not a resumable training directory;
start a fresh reproduction to recompute it.

Deleting generated data and results
-----------------------------------

To reset a stopped benchmark to the files needed for a fresh run, first preview:

    python delete_population_pertubation_train_convergence_data.py --dry-run

Then perform the deletion:

    python delete_population_pertubation_train_convergence_data.py

The default target is the folder containing the script, even from another
working directory. An optional positional FOLDER targets a separate benchmark
copy. Without --dry-run, deletion is immediate. Stop the benchmark first; the
script refuses a reset while an existing launcher, manager or report lock is held.

This reset deletes all saved results, prepared arrays, downloaded sources,
checkpoints, reports, figures, logs, generated verification records, runtime
statuses (including FINALISATION.json), environment_runs.json, REPRODUCTION.json,
CLEANUP.json, caches, temporary files and other files outside its keep rules.
Unknown files inside the target are also deleted. It writes no new cleanup log.

It retains Python/shell scripts, Python tests, frozen source/patches, requirements,
README.txt, protocol.json, environment_original.json, HOLDS.json, Git metadata and
the bundled Titanic/New Thyroid CSVs. It also retains the authoritative manifest
for each dataset (reference_manifests/ when present, otherwise data/*/manifest.json)
and verification/source_split_overlap.json. These are bootstrap inputs required
by reproduce.py; deleting them would prevent a fresh launch. Required inputs and
their pinned source hashes are checked before any deletion. Symlinked directories
are not followed, and deleting an unneeded link leaves its target untouched.

After resetting, run python reproduce.py without --resume. Sources and arrays
will be rebuilt and all models will start fresh. The separate
clean_population_pertubation_train_convergence_data.py command remains available
for archival cleanup that retains results, reports, logs, verification evidence,
REPRODUCTION.json, environment_runs.json, configuration, manifests and bundled
inputs. It preserves and updates CLEANUP.json. It removes transient runtime state,
including top-level FINALISATION.json and the exact path
verification/latest_monitoring_check.json.

Sources and environment
-----------------------

Hugging Face inputs are downloaded from the exact 40-character revisions in
the archived manifests and verified against recorded byte counts and SHA-256.
HIGGS is downloaded from its recorded UCI URL with its recorded checksum.
Covertype uses the pinned scikit-learn downloader (which verifies the upstream
archive checksum); its final arrays must also match all archived split hashes.
The exact original Titanic CSV (OpenML 40945) and New Thyroid CSV are small and
bundled in inputs/; both are checked against their original source hashes. This
avoids external sibling directories or user-specific paths. Keep inputs/ in Git.

requirements.txt pins the installed dependency closure for the benchmark,
including PyTorch/CUDA packages. environment_original.json records the original
Python 3.13.2 environment. The documented portable installation uses Python 3.12
because NumPy 1.26.4 has published wheels for that version. The package versions
are unchanged; this is not a claim of an identical interpreter build or guaranteed
bitwise-identical model results across machines. Array hashes and the numerical
checks enforce the relevant data/backend checks. environment_runs.json records
the actual environment on each launch. A mismatched pinned dependency fails early.

Historical absolute paths in protocol.json and reference manifests are provenance
only; they are not used to locate runtime data. source_hashes in protocol.json
verify the eight frozen production modules. The frozen ANNpt_data.py and
ANNpt_globalDefs.py include the bank-full loader/configuration update; the other
modules include the selectable loss controller and its configuration; the batched
GPU estimator is unchanged.

Banking Marketing policy
------------------------

Banking Marketing now reads only bank-full.csv from the official UCI archive:

    https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip

The archive size and SHA-256 are pinned in data/banking-marketing/manifest.json.
All 45,211 rows are used exactly once. The existing single-source stratified
60/20/20 partition function, seed 20260930, produces 27,126 training, 9,042
validation and 9,043 test rows. Categories and normalization remain fitted only
on training rows. Exact full-record comparisons show zero overlap between
these partitions; see verification/source_split_overlap.json.

bank.csv is a 10% sample of bank-full.csv, not an independent test set. The old
Hugging Face train/test combination loaded both and duplicated every supplied
test record. It is no longer used. The datasetName remains banking-marketing;
bank-additional-full.csv is a different dataset and is not used by this patch.

The production loader also reads only bank-full.csv, then uses its existing
datasetTestSplitSize (default 0.1) to create train/test splits: 40,689/4,522 rows.
Its split policy is therefore still different from this benchmark's 60/20/20.

This change revises the experiment protocol and prepared-data hashes. Start a
fresh launch with python reproduce.py (or a new --output folder) to use it.
Old overlapping data/checkpoints must not be resumed under the new protocol.
Previously created independent output folders keep their own frozen code; they
must be replaced by a fresh reproduction from this updated folder to use bank-full.
--resume continues only runs initialised with the updated code and protocol.

Archive and validation
----------------------

Existing REPORT.txt, runs/*.json, figures and per-run verification files are
replaced when reproducing in this folder. REPRODUCTION.json hashes the snapshot,
reference_manifests/ retains expected arrays/metadata, and checkpoints/ contains
that reproduction's working state. Recomputed reports are written only after
real training; nothing is presented as a rerun merely by copying saved results.
The manager regenerates verification gates, independently evaluates saved best
models, checks paired initialization/minibatch streams, and validates figure
files. Final scientific visual review remains a separate human review step.

The saved reports/results and earlier verification records predate the bank-full
change; they remain historical records, not results of the revised experiment.
report.py rejects results carrying a different protocol instead of relabelling
them as new measurements. No training was performed as part of the dataset patch.

Run the focused portability tests (no GPU training or network needed):

    python -m unittest discover -s tests -v

Validation evidence for this portability patch is saved in
verification/portability/checks.json and its companion records. All 13 complete
datasets were rebuilt and audited in the original environment. A separately
installed Python 3.12 environment passed dependency checks, 11 portability tests,
full Iris/Titanic/New Thyroid reconstruction, preprocessing equivalence, and
numerical fixtures for all four population sizes, including CUDA graph/RNG
resume checks. This validation did not launch the 195-run training experiment.

The subsequent in-place launcher change passed all 17 focused tests and real
preparation/audit launches on disposable copies using complete Titanic and New
Thyroid datasets. Fresh default launches replaced saved-result/checkpoint
fixtures, --resume preserved them, and separate --output launches still worked.
See verification/portability/in_place_launch.json for this follow-up evidence.

The bank-full update is separately validated in
verification/portability/bank_full_upgrade.json. The earlier portability records
describe their original source versions and do not validate the revised bank
splits; the new record covers full-source loading, production preprocessing,
disjoint splits, new prepared-array hashes, and rejection of the old data/protocol.
