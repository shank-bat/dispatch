-- Dispatch initial schema.
--
-- Design notes live in docs/ARCHITECTURE.md §5. The short version:
--   * The daemon is the only writer, so there is no contention to design around.
--   * States are TEXT, not integers, because a `sqlite3` session at 2am over SSH should
--     show RUNNING rather than 3.
--   * Partial indexes keep the scheduler's hot query on a handful of rows no matter how
--     large history grows.
--   * `jobs.metadata` JSON is canonical; `job_metadata` and `jobs_fts` are derived
--     indexes maintained alongside it, because json_extract() in a WHERE clause cannot
--     use an index.

CREATE TABLE jobs (
    id                TEXT    PRIMARY KEY,
    seq               INTEGER NOT NULL UNIQUE,
    name              TEXT    NOT NULL,
    workdir           TEXT    NOT NULL,
    solver            TEXT    NOT NULL,
    solver_binary     TEXT,
    cores             INTEGER NOT NULL CHECK (cores >= 1),
    ram_estimate_mb   INTEGER CHECK (ram_estimate_mb IS NULL OR ram_estimate_mb > 0),
    priority          INTEGER NOT NULL DEFAULT 0,
    state             TEXT    NOT NULL CHECK (state IN (
                          'QUEUED','HELD','PREPARING','RUNNING',
                          'COMPLETED','FAILED','CANCELLED','REJECTED','UNKNOWN')),
    created_at        REAL    NOT NULL,
    started_at        REAL,
    finished_at       REAL,
    exit_code         INTEGER,
    exit_reason       TEXT,
    exit_signal       TEXT,
    stdout_path       TEXT,
    stderr_path       TEXT,
    pid               INTEGER,
    pid_start_time    REAL,
    metadata          TEXT    NOT NULL DEFAULT '{}',

    -- Measured metrics, filled progressively while the job runs (§6.6).
    runtime_s         REAL,
    peak_rss_mb       INTEGER,
    mean_cpu_pct      REAL,

    CHECK (json_valid(metadata))
);

-- The scheduler's hot query. Partial, so it indexes only the queue -- typically tens of
-- rows -- and never grows with completed history.
CREATE INDEX idx_jobs_queue  ON jobs (priority DESC, seq ASC) WHERE state = 'QUEUED';
CREATE INDEX idx_jobs_active ON jobs (state)                  WHERE state IN ('PREPARING','RUNNING');
CREATE INDEX idx_jobs_recent ON jobs (finished_at DESC)       WHERE finished_at IS NOT NULL;
CREATE INDEX idx_jobs_created ON jobs (created_at DESC);
CREATE INDEX idx_jobs_solver ON jobs (solver);

-- Append-only audit trail. Every state change writes one row, so "what happened to this
-- job, in order" stays answerable for jobs that finished months ago.
CREATE TABLE job_events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id   TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,
    detail   TEXT NOT NULL
);
CREATE INDEX idx_events_job ON job_events (job_id, id);

CREATE TABLE notes (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id   TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ts       REAL NOT NULL,
    body     TEXT NOT NULL
);
CREATE INDEX idx_notes_job ON notes (job_id, id);

-- Coarse resource time series for running jobs.
CREATE TABLE job_samples (
    job_id   TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ts       REAL NOT NULL,
    rss_mb   INTEGER NOT NULL,
    cpu_pct  REAL NOT NULL,
    PRIMARY KEY (job_id, ts)
) WITHOUT ROWID;

-- Tags (§4.5), normalised so renaming one is a single UPDATE.
CREATE TABLE tags (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name  TEXT NOT NULL UNIQUE
);
CREATE TABLE job_tags (
    job_id  TEXT    NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    tag_id  INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (job_id, tag_id)
) WITHOUT ROWID;
CREATE INDEX idx_job_tags_tag ON job_tags (tag_id, job_id);

-- Derived typed index over declared metadata fields (§4.4, §13.18). Exactly one of
-- value_text / value_num is populated per row, so `endTime > 500` compares as a number.
CREATE TABLE job_metadata (
    job_id     TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value_text TEXT,
    value_num  REAL,
    PRIMARY KEY (job_id, key)
) WITHOUT ROWID;
CREATE INDEX idx_meta_key_num  ON job_metadata (key, value_num)  WHERE value_num  IS NOT NULL;
CREATE INDEX idx_meta_key_text ON job_metadata (key, value_text) WHERE value_text IS NOT NULL;

-- Reproducibility record (§6.9). Written once, at job start.
CREATE TABLE job_provenance (
    job_id           TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    captured_at      REAL NOT NULL,
    dispatch_version TEXT NOT NULL,
    python_version   TEXT NOT NULL,
    hostname         TEXT NOT NULL,
    kernel_version   TEXT NOT NULL,
    os_release       TEXT,
    cpu_model        TEXT,
    total_ram_mb     INTEGER,
    solver_name      TEXT,
    solver_version   TEXT,
    adapter_version  INTEGER,
    git_commit       TEXT,
    git_branch       TEXT,
    git_dirty        INTEGER,
    git_remote       TEXT,
    env_snapshot     TEXT NOT NULL DEFAULT '{}',
    env_hash         TEXT,
    argv             TEXT NOT NULL DEFAULT '[]',
    CHECK (json_valid(env_snapshot) AND json_valid(argv))
);
CREATE INDEX idx_prov_dirty ON job_provenance (git_dirty) WHERE git_dirty = 1;

-- Full-text search. Contentless-external would save space but complicates rebuilds; at a
-- few thousand rows the duplicated text is well under a megabyte.
CREATE VIRTUAL TABLE jobs_fts USING fts5(
    job_id UNINDEXED,
    name,
    workdir,
    solver,
    solver_binary,
    metadata,
    notes,
    tags,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- The FTS row is rebuilt wholesale by the repository whenever any contributing part
-- changes (name, metadata, notes, tags). Triggers on `jobs` alone could not see notes or
-- tags, and a trigger web spanning four tables would be harder to reason about than one
-- explicit call at the single point where writes happen.
