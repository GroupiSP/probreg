# CMAPSS Example Data

Loading the NASA CMAPSS FD001 dataset for the JAX example pipeline (`examples/jax/cmapss/`): fetching it from NASA's servers, keeping a local copy, and letting a user supply their own copy when the network can't be trusted.

## Language

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
