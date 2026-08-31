-- CPU/GPU resource classification, and the solver log's home in the case directory.
--
-- Three columns, all nullable-or-defaulted, so every row written before this migration
-- keeps its exact previous meaning:
--
--   resource_kind  'cpu' for every existing job, which is what they all were. The kind is
--                  stored rather than derived from `gpus > 0` because the two answer
--                  different questions: how many GPUs a job holds, and which pool it was
--                  admitted from. A GPU job that asked for one GPU and eight cores is not
--                  the same thing as an eight-core CPU job, and a search for GPU work
--                  should find it.
--
--   gpus           0 for every existing job. CHECKed against resource_kind so the pair
--                  cannot drift apart in the database any more than it can in the model:
--                  a 'cpu' row holds no GPUs, a 'gpu' row holds at least one.
--
--   log_path       NULL for every existing job. Their output is where it has always been,
--                  under the per-job log directory, and `stdout_path` still points at it --
--                  which is why upgrading does not make a single historical log
--                  unreadable. New jobs additionally record the working-directory log
--                  here, and set stdout_path to the same file.

ALTER TABLE jobs ADD COLUMN resource_kind TEXT NOT NULL DEFAULT 'cpu'
    CHECK (resource_kind IN ('cpu', 'gpu'));

ALTER TABLE jobs ADD COLUMN gpus INTEGER NOT NULL DEFAULT 0
    CHECK (gpus >= 0
           AND (resource_kind = 'gpu') = (gpus > 0));

ALTER TABLE jobs ADD COLUMN log_path TEXT;

-- Partial, like every other scheduling index here: GPU jobs are the minority, so the
-- index stays proportional to them rather than to the whole history.
CREATE INDEX idx_jobs_gpu ON jobs (state, gpus) WHERE gpus > 0;
