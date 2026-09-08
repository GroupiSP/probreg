# probreg

The shared vocabulary of this repository: the language of a training run and of recording one with an experiment tracker, which the library itself declares, and the language of the NASA CMAPSS FD001 remaining-useful-life example — obtaining the dataset, and describing the targets the models predict and the curves that visualize them.

One section per area. A term belongs in the training-and-tracking section when the library uses it, and in the CMAPSS section when only that example does.

## Training and tracking

**Training event**:
A structured observation emitted at a named point in a stage's lifecycle, carrying that stage's metrics and the live training state. Not a log line: it has no format, and what it means is decided by whoever consumes it.
_Avoid_: Log record, message

**Event sink**:
A consumer notified of every training event. Passive: it observes a run and cannot alter it, so a runner never waits on what a sink returns.
_Avoid_: Callback (implies the runner awaits a decision), logger, hook

**Experiment tracker**:
A destination that records parameters, metrics and artifacts for one run. Ignorant of probreg's event model: it is told what to record, never when a run reaches a point of interest.
_Avoid_: Logger, writer, backend

**Run**:
One execution of a training script, and the unit a tracker groups its records under. One tracked run maps to one TensorBoard run directory.
_Avoid_: Experiment (a set of runs), trial

## CMAPSS example

**Archive cache**:
The persistent, on-disk copy of the downloaded CMAPSS zip, kept across process runs so the dataset is fetched from NASA at most once rather than once per loader call or per run.
_Avoid_: Temp directory, download folder

**Cache directory**:
The directory holding the archive cache, resolved from `PROBREG_CMAPSS_CACHE_DIR` if set, otherwise a stdlib-computed per-user cache location (honoring `XDG_CACHE_HOME`, else `~/.cache/probreg-cmapss`).

**Archive override**:
A user-supplied path to a manually obtained copy of the CMAPSS zip, provided via the `PROBREG_CMAPSS_ARCHIVE` environment variable or an optional loader parameter. When present, it seeds the archive cache and no download is attempted.
_Avoid_: Manual download, local copy

**Fetching**:
Obtaining the CMAPSS archive and placing it in the archive cache — via network download (with retries) or via an archive override. Distinct from *loading*, which reads sensor columns out of an archive already in the cache.
_Avoid_: Downloading (too narrow — excludes the override path)

**Loading**:
Reading a specific split (train, test, or ground-truth RUL) out of the archive cache into a DataFrame/array. Assumes fetching has already happened, and triggers it transparently if it hasn't.

**Lifetime**:
A run-to-failure unit's total number of cycles, i.e. its remaining useful life at cycle 0 — `max(time_cycles)`. Defined only for units from the train split, whose trajectories run to failure; a truncated test trajectory has no observable lifetime, only a ground-truth remaining life at its last observed cycle.
_Avoid_: Duration, age, total RUL

**Linear RUL**:
The unclipped remaining-useful-life target `max(time_cycles) - t`, decreasing by exactly one per cycle. This is what the example's models are trained on, as opposed to the widely used piecewise-constant convention that caps RUL at a constant (typically 125 or 130) early in life. Naming it explicitly keeps a reader from assuming the capped convention and misreading the targets and curves.
_Avoid_: RUL (ambiguous between the two conventions), piecewise RUL

**Full window**:
A window whose every row is an observed cycle of the unit, as opposed to a *padded window*, whose leading rows repeat the unit's first observed reading to make a short trajectory reach the window length. A padded window's remaining-useful-life target is still computed from real cycles, so it is a usable training sample; but its sensor history is fabricated, so a prediction made from it has no support in the data.
_Avoid_: Complete window, valid window, real window

**RUL curve**:
For a single unit, its linear RUL and the model's predictive RUL plotted against time cycles. The predicted curve begins at the unit's first full window, since no prediction exists before `window_length` cycles of history have accumulated.
_Avoid_: RUL plot, degradation curve

**Lifetime-spanning units**:
The shortest-, lower-median-, and longest-lifetime units of a set of run-to-failure trajectories, drawn side by side so that a reader can tell whether the model's accuracy and its stated uncertainty behave consistently across units that fail early and units that survive several times as long. Ordered ascending by lifetime with the unit ID as tie-break, and the median is the lower of the two central entries, so the same data always yields the same three units.
_Avoid_: Representative units, extreme units, best/worst units (none of the three is chosen for how well the model does on it)

**Predictive interval**:
The `loc ± 1.96 * scale` band of the composite Gaussian predictive distribution, covering 95% of its mass and left unclipped at zero, so that a band extending below zero remains visible as evidence of the Gaussian assumption breaking down near end of life.
_Avoid_: Confidence interval (this is an interval over a predicted value, not over a parameter estimate)
