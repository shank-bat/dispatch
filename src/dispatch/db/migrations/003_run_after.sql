-- "Run this job only after that one finishes."
--
-- One nullable self-reference on `jobs`, and nothing else. A dependency is a property of
-- the job that asked for one, so it belongs on the job row rather than in a table of
-- edges: there is no graph here, no fan-in, and no second consumer of the relationship.
--
-- NULL means what it has always meant -- schedule this job as soon as it fits -- so every
-- row written before this migration keeps its exact previous behaviour.
--
-- ON DELETE SET NULL rather than CASCADE or RESTRICT: deleting a finished job must not
-- delete the job that waited for it, and must not start failing because something once
-- named it. Clearing the reference leaves the dependent job schedulable, which is the
-- same position it would have been in had the parent never been named.

ALTER TABLE jobs ADD COLUMN depends_on_job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL;

-- Partial, like the other scheduling indexes: dependencies are rare, so the index stays
-- proportional to the jobs that have one rather than to history.
CREATE INDEX idx_jobs_depends_on ON jobs (depends_on_job_id) WHERE depends_on_job_id IS NOT NULL;
