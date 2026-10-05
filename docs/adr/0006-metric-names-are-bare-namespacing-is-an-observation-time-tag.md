# Metric names are bare; namespacing is an observation-time tag

A metric used to be renamed by every layer it passed through. `HeldOutValidation` prefixed `validation_`, the runner prefixed `training_`, the staged runner prepended the stage name with `_`, and `TrackerEventSink` rebuilt a `/`-joined tag from the event name. That produced four coexisting schemes and three spellings of the training split (`train`, `training`, `train/`), and a history key such as `mean_training_loss` that cannot be split back into its parts (issue #34). We decided that a metric has exactly one name: bare `snake_case`, chosen by whoever computes it. Namespacing is not part of that name. It is a **metric tag**, `stage/split/metric`, built only where a run is observed, with `/` as the only separator and no segment allowed to contain it. `probreg.core.naming` owns the separator, the `Split` vocabulary (`train`, `validation`), and the functions that build and parse tags and parameter paths. No other code joins or splits these strings.

The split moves from guesswork into data. `TrainingEvent` carries a required `split`, set by the runner, which is the only party that knows it. The sink no longer infers the split from the event's name. `state.metric_history` is keyed by the same tag string the tracker shows, so a history key and a TensorBoard tag for the same series are the same string. Events that report a decision rather than a measurement (`best_model`, `early_stop`) carry no metrics. Their monitored metric and value travel in `payload` instead. Otherwise, tagging every event unconditionally would log each improving epoch's point twice.

## What this amends in ADR 0005

ADR 0005 stands in its main decision: core ships the bridge to trackers, never a tracker, and `TrackerEventSink` remains that bridge. The following parts are amended:

- **Who names the tag scheme.** ADR 0005 says the adapter "names the tag scheme". The scheme now belongs to `probreg.core.naming`, and `TrackerEventSink` is one of its callers, alongside the runners that write `state.metric_history`. The sink no longer owns any naming logic.
- **How the tag is built.** The ADR 0005 tag was `<stage>/<event prefix><metric>`, where the prefix came from the event name via `DEFAULT_EVENT_PREFIXES` and the sink's configurable `event_prefixes`. Events missing from that mapping were logged straight under `<stage>/`. Both the mapping and the field are removed. The tag is now always `metric_tag(event.stage, event.split, name)`, and a new event needs no registration.
- **The stage segment is still unconditional.** ADR 0005's argument (a tag without the stage lets one stage's curve overwrite another's) now applies to `state.metric_history` too, including single-stage runs.
- **Parameters.** ADR 0005 left `log_params` entirely to the caller, and the path syntax was decided in the TensorBoard example. Calling `log_params` is still the caller's job, and no training event carries parameters. But the *parameter path* syntax (a nested mapping flattened to `/`-joined key paths, with the split as its own segment) now lives in `probreg.core.naming`, beside the metric tag. A tracker receives the nested mapping and flattens it with the core helper. Only vendor-specific value coercion, such as stringifying what HParams cannot store, stays in the example. The decision moves into core and the dependency stays out, which is the same division ADR 0005 drew.

## Considered Options

- **Structured history keys** (a `(stage, split, metric)` tuple or nested mapping), with the tag string built only at the tracker. With this, nothing would ever be parsed. Rejected: history and tracker would then show the same series under different names. A strict `parse_metric_tag` is safe because no segment may contain the separator.
- **Keep per-layer prefixes but make them consistent.** Rejected: any layer that prefixes a name it did not compute is a naming authority that can drift. That drift is the bug: early-stopping configuration had to match whatever `metric_prefix` was set to, and `_latest_training_metrics` silently found nothing when the prefixes disagreed.
- **Keep `best_model` / `early_stop` metrics and have the sink skip them by event name.** Rejected: it brings back an event registry in the sink, which is what removing `DEFAULT_EVENT_PREFIXES` set out to avoid.

## Consequences

This breaks anyone reading `state.metric_history` keys, configuring `metric_prefix`, `metric_history_prefix` or `event_prefixes`, or using `MetricSource`, which `Split` replaces. TensorBoard tag paths change too, so runs written before the change will not line up with runs written after it. `EarlyStopper.metric` is always a bare name, and `source` alone selects the split.
