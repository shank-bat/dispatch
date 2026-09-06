-- Sweeps, and the flag that survives a reboot.
--
-- Two independent features, one migration, because they are one schema revision and
-- splitting them would only make the version numbers lie about what a database at
-- version 5 contains.
--
--
-- SWEEPS
--
-- A sweep is a scheduling group, not a job: every member is an ordinary row in `jobs`
-- holding its own ordinary allocation. All a sweep adds is one rule -- at most
-- `concurrency` of its members may run at once -- so all it needs is somewhere to put
-- that number and a way for a job to name it.
--
-- Deliberately NOT modelled as a parent job with children. A sweep never runs, never has
-- an exit code, never holds cores, and never appears in the queue; giving it a row in
-- `jobs` would mean every query in the scheduler having to exclude it.
--
-- `cores_per_job` is stored even though each member's own `cores` column already holds
-- it. That is not duplication of the scheduling input -- the member's column is what the
-- scheduler reads and remains authoritative -- it is the sweep's *configuration*, which
-- has to survive independently so that the settings can still be shown, and reasoned
-- about, after members have been deleted from history.

CREATE TABLE sweeps (
    id             TEXT    PRIMARY KEY,
    name           TEXT    NOT NULL,
    root           TEXT    NOT NULL,
    solver         TEXT    NOT NULL,
    cores_per_job  INTEGER NOT NULL CHECK (cores_per_job >= 1),

    -- The hard cap. CHECKed at >= 1 because a sweep that may never run anything is not a
    -- configuration anyone means to express, and would sit in the queue forever.
    concurrency    INTEGER NOT NULL CHECK (concurrency >= 1),

    created_at     REAL    NOT NULL
);

-- NULL for every job that existed before this migration, and for every ordinary
-- submission afterwards. That is what makes the feature additive: the scheduler's sweep
-- rule reads this column, finds NULL, and leaves the job exactly as opportunistic as it
-- has always been.
--
-- ON DELETE SET NULL, matching depends_on_job_id: deleting a sweep must release its
-- members to ordinary scheduling, never delete them and never start failing.
ALTER TABLE jobs ADD COLUMN sweep_id TEXT REFERENCES sweeps(id) ON DELETE SET NULL;

-- Position within the sweep, from zero, in the order the cases were found on disk.
-- Queue order still comes from `seq` -- members are created in this order, so the two
-- agree -- but `seq` is global and cannot answer "which case of the sweep is this?".
ALTER TABLE jobs ADD COLUMN sweep_position INTEGER;

-- Partial: sweep members are a minority of history, so the index that the scheduler's
-- per-sweep running count uses stays proportional to sweeps rather than to every job
-- ever run.
CREATE INDEX idx_jobs_sweep ON jobs (sweep_id, sweep_position) WHERE sweep_id IS NOT NULL;


-- RESUME AFTER REBOOT
--
-- Set when startup recovery finds a job that was RUNNING before the machine last booted:
-- its process cannot exist, so the job returns to the queue to be restarted from whatever
-- state the simulation itself last wrote.
--
-- A boolean, and only a boolean. It records that a resume is wanted; it deliberately does
-- NOT record a timestep, an iteration, or a checkpoint path. Dispatch does not know what
-- a simulation finished writing -- only the simulation's own files do -- so the restart
-- point is read from the case by the adapter at plan time. Storing a number here would
-- make Dispatch's database an authority on something it cannot observe, and the first
-- time the two disagreed it would resume a run from a timestep that was never written.
--
-- 0 for every existing row: no job written before this migration was interrupted by a
-- reboot this daemon knows about, and claiming otherwise would restart finished work.
ALTER TABLE jobs ADD COLUMN resume_requested INTEGER NOT NULL DEFAULT 0
    CHECK (resume_requested IN (0, 1));

-- Which boot of this machine the job started on, as the kernel's own `btime`.
--
-- This is how a reboot is told apart from a daemon restart, and it is a recorded fact
-- rather than an inference. The tempting comparison -- "did the machine boot after the job
-- started?" -- silently assumes the job's timestamps and the system clock share an epoch,
-- which is true in production and quietly false anywhere the clock is injected. Comparing
-- a boot identity against the same boot identity has no such assumption: either it is the
-- boot the job started on or it is not.
--
-- NULL for every existing row, and for any machine that cannot report `btime`. A job whose
-- boot is unknown is NOT treated as rebooted: it keeps the pre-existing `lost` outcome,
-- because restarting a simulation on no evidence is worse than admitting ignorance.
ALTER TABLE jobs ADD COLUMN boot_time REAL;
