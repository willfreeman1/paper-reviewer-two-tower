# Frozen scores

`scoreboard.json` is the official public table. `README.md` and
`docs/how-it-works.md` must match it.

`eval_snapshots/` are the raw files those rows were read from (copied
out of `spike/data/`, which stays off GitHub because it is huge).
Local Windows paths in those copies were shortened to repo-relative
paths.

Do not edit the snapshots to “fix” a number. If a re-run disagrees,
change `scoreboard.json` only after the code check says the new run is
right.
