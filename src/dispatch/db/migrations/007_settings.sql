-- Settings changed from the interface rather than in config.toml.
--
-- The first is the CPU counting mode (§4.3.2), which the dashboard can now switch while the
-- daemon runs. It has to survive a restart -- a mode that silently reverted after a reboot
-- would mean the machine scheduling differently from what its own dashboard last said --
-- and it cannot be written back into config.toml, because rewriting a file the user edits
-- by hand would discard their comments and their formatting, and racing their editor.
--
-- So the daemon keeps it here, in the database it already owns. Precedence is explicit and
-- reported: a row here overrides the config file's value until it is cleared, and setting a
-- value equal to the config file's clears the row rather than duplicating it, so "back to
-- what the file says" is always one keypress.
--
-- Key/value rather than a column per setting because the set is small, read once at startup,
-- and likely to grow; a migration per toggle would be ceremony for nothing.
CREATE TABLE settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
