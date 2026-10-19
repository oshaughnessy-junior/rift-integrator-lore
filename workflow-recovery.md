# Recovering RIFT workflows without repeating completed work

Reusable operating guidance for ILE/CIP campaigns. Host paths, scheduler policy,
container availability, and storage rules belong in infra-atlas. Campaign job
identities, budgets, and receipts belong in the campaign records.

## Before launching or changing a workflow

Read both local and target-host project memory, the relevant run recipes, RIFT
orientation, and infra-atlas. Inspect the live service metadata and tool hashes
before copying scripts: another session may have deployed a newer service.
Record the source commit, input/configuration pins, and generated arguments.

A running ILE queue does not mean the whole DAG can advance. Read failed/futile
node counts as well as completed nodes and actual likelihood products. Keep
routine monitoring and known operational repair in local scripts; reserve human
or LLM review for unknown failures and scientific milestones.

## Native rescue and master DAGs

After a normal failed DAG exits, verify that its children have also left the
queue. Inspect the newest native rescue file and its failed-node evidence.
Preserve completed nodes and existing likelihood products. Do not launch a
second DAG over a still-active workflow, or mark an ILE node DONE because it
produced only part of its assigned grid.

A bounded master DAG can retry a subworkflow from its native rescue. This is a
template; replace both paths with the actual run paths:

```
SUBDAG EXTERNAL RIFT /path/to/run/the.dag DIR /path/to/run
RETRY RIFT 3
```

RIFT's `bin/util_ConsolidateDAGsUnderMaster.sh` emits this pattern. Its recursive
`find` also discovers runtime CIP sub-DAGs: explicitly select the intended
root workflows instead of wrapping every `.dag` beneath a run directory.
Retain the science sub-DAG's concurrency limits and its intended executable
PATH/source environment when resuming.

An external repair script that submits the same native rescue with a finite
budget is another valid implementation. Journal intent before submitting and
use a unique scheduler tag to recover an uncertain submission result. If it is
ambiguous whether submission succeeded, stop rather than submit again.

Gate retries on diagnosed operational causes and unchanged scientific pins.
Respect explicit disabling, user/site holds, removal, and exhausted budgets.
Unknown application errors or scientific invariant failures need review.

A successful scheduler exit is insufficient evidence of convergence. Some RIFT
convergence nodes deliberately abort their DAG with a success return code;
crashes can take the same path. Inspect the actual convergence verdict and final
products. See the convergence-test entry in [gotchas.md](gotchas.md).

## Preserve partial work and bound worker repairs

- Preserve per-point ILE outputs across eviction/removal. With file transfer,
  check the actual transfer/checkpoint policy; output returned only on normal
  exit can lose completed points. Reuse valid points only with compatible source,
  waveform, inputs, priors, and likelihood conventions.
- Count distinct physical points separately from evaluation rows. Retries and
  repeated fair draws can duplicate points; file/row counts alone overstate
  coverage.
- Size worker batches using measured point costs and available uninterrupted
  runtime. Larger chunks can reduce delivered throughput when repeated eviction
  discards the whole batch.
- Correct submit files as well as existing job ads. DAGMan reads submit files for
  future nodes, and CIP sub-DAGs can generate additional files at runtime.
- A `qedit` readback proves an ad changed, not that a running job adopted its new
  walltime policy. CIT has enforced original limits after a successful readback.
  Preserve those attempts and resume from corrected submit files.
- Match failure evidence to the campaign's owner and DAG before repair. A
  verified missing container mount despite an advertised capability can justify
  a recorded, scoped worker exclusion. Use `TARGET.Machine` in requirements;
  unqualified `Machine` can resolve against the job ad instead.
- Journal operational changes and their old/new hashes. Never refresh scientific
  pins to make an unexpected change pass validation. Verify that requirements or
  resource amendments leave scientific arguments unchanged.

For fast intermediate development steps, a failed join/conversion can sometimes
be executed by hand. Run the exact step in the correct environment and validate
its output before marking that node DONE in the latest rescue. This is not a
substitute for recovering missing likelihood evaluations.

## Evidence and provenance

The 2026-10-17 zero-spin campaign used RIFT O4d commit
`d8f676e4a30b24e816f26085f21bc90c518d95b6`. Its
[operational amendments](https://github.com/oshaughnessy-junior/RIFT_roboto_paper/tree/1f31bfcd/analyses/eccentric_zero_spin_20261017/records)
retain missing-mount failures, requirements-only changes, walltime corrections,
and original pins. These are infrastructure observations, not posterior results.

A separate harmless
[master-DAG probe](https://github.com/oshaughnessy-junior/RIFT_roboto_paper/tree/04062608/analyses/eccentric_zero_spin_20261017/records/automation_native_probe)
verified that a completed node ran once while a deliberately failed node ran
twice and the master exited successfully. The same record contains the deployed
[autonomous repair implementation and tests](https://github.com/oshaughnessy-junior/RIFT_roboto_paper/tree/04062608/analyses/eccentric_zero_spin_20261017/records/autonomous_followup_deployed/tools).
These tests establish recovery mechanics, not scientific correctness.
