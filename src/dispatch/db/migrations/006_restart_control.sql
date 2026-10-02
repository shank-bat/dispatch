-- Two ways a run's starting point stops being "the beginning", and the core count that
-- may change between one attempt and the next.
--
-- One migration, because they are one schema revision and splitting them would only make
-- the version numbers lie about what a database at version 6 contains. They also share a
-- mechanism: both end in a job being planned with "continue from whatever the simulation
-- itself last wrote" rather than from the case's configured start time, which the adapter
-- already knew how to do for a reboot resume (§6.7).
--
--
-- START FROM THE LATEST WRITTEN TIME
--
-- An explicit, sticky request to continue a case from its own most recent output instead
-- of its configured start. Distinct from `resume_requested`, which it would have been
-- tempting to reuse, for two reasons:
--
--   * They mean different things to a reader. `resume_requested` records an accident --
--     "the machine rebooted under this job" -- and the audit trail says so. A sweep
--     configured to continue its cases is a deliberate setting, and a history that
--     reported it as a reboot would be fiction.
--   * They have different lifetimes. `resume_requested` is transient and cleared the
--     moment the restart it asked for has been planned, so it cannot cause a second one.
--     This is a property of how the job was configured and outlives any number of
--     restarts: a case asked to continue should still continue after a reboot.
--
-- 0 for every existing row, so every job already in the database keeps starting exactly
-- where it always did.
ALTER TABLE jobs ADD COLUMN start_from_latest INTEGER NOT NULL DEFAULT 0
    CHECK (start_from_latest IN (0, 1));

-- The sweep's own copy of the setting, stored for the same reason `cores_per_job` is: the
-- member column is what the planner reads and stays authoritative, while this is the
-- sweep's *configuration*, which has to remain answerable after members have been deleted
-- from history.
ALTER TABLE sweeps ADD COLUMN start_from_latest INTEGER NOT NULL DEFAULT 0
    CHECK (start_from_latest IN (0, 1));


-- PENDING CORE CHANGE
--
-- How many cores a running job should come back with after it has been asked to stop at
-- its next write. NULL means no change is pending, which is every job that has never been
-- repartitioned and every row written before this migration.
--
-- A pending *request* rather than a new state, and that is the whole design. The job stays
-- RUNNING while the solver finishes its timestep, then takes the RUNNING -> QUEUED edge
-- that reboot resume already established (§6.7) with its `cores` column rewritten in the
-- same statement. So there is no PAUSED state to teach the scheduler, no second kind of
-- allocation for the ledger to track, and a job mid-repartition is -- correctly -- either
-- a running job or a queued one at every instant.
--
-- Stored rather than held in memory because the daemon may restart between the request and
-- the solver noticing it. A request that lived only in the executor would be lost, and the
-- job would quietly finish on its old core count having been told it was changing.
--
-- CHECKed at >= 1: a repartition to zero cores is not a configuration anyone means to
-- express, and would queue a job that can never be admitted.
ALTER TABLE jobs ADD COLUMN repartition_cores INTEGER
    CHECK (repartition_cores IS NULL OR repartition_cores >= 1);

-- Partial, like the other scheduling indexes: a pending repartition is rare, so the index
-- the executor's "is a change waiting for this job?" lookup uses stays proportional to
-- the requests rather than to every job ever run.
CREATE INDEX idx_jobs_repartition ON jobs (repartition_cores)
    WHERE repartition_cores IS NOT NULL;
