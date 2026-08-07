-- Why a job failed, in the solver's own words.
--
-- The exit code says a run failed; it never says why, and the answer is always sitting in
-- the log the user then has to go and open. This column holds the few lines that actually
-- explain it, extracted when the job reaches a terminal state (§6.6).
--
-- Nullable, and empty for every job that succeeded: there is nothing to explain about a
-- run that worked. Also empty for older rows, which is correct -- their logs were never
-- read for this and inventing an explanation after the fact would be worse than a blank.

ALTER TABLE jobs ADD COLUMN exit_detail TEXT;
