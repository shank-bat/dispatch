"""Configuration loading.

The rejection tests carry the weight here. A typo in ``config.toml`` that is silently
ignored means a machine behaving differently from what its configuration says, discovered
months later while wondering why a large job never ran.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dispatch.core.config import Config, SchedulerConfig, load_config
from dispatch.core.errors import ConfigError


def write(tmp_path: Path, body: str) -> Path:
    target = tmp_path / "config.toml"
    target.write_text(body)
    return target


def test_defaults_work_with_no_file_at_all(tmp_path: Path) -> None:
    """Dispatch must run before the user has written any configuration."""
    config = load_config(tmp_path / "absent.toml")
    assert config.scheduler.policy == "backfill"
    assert config.scheduler.reserved_cores == 1
    assert config.daemon.autostart is True
    assert config.source is None


def test_a_missing_file_can_be_made_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml", required=True)


def test_values_override_defaults(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [scheduler]
        policy = "fifo"
        reserved_cores = 4
        ram_margin_mb = 8192

        [daemon]
        autostart = false
        cancel_grace_s = 30.0
        """,
    )
    config = load_config(path)
    assert config.scheduler.policy == "fifo"
    assert config.scheduler.reserved_cores == 4
    assert config.scheduler.ram_margin_mb == 8192
    assert config.daemon.autostart is False
    assert config.daemon.cancel_grace_s == 30.0
    assert config.source == path


def test_unspecified_values_keep_their_defaults(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, "[scheduler]\nreserved_cores = 2\n"))
    assert config.scheduler.reserved_cores == 2
    assert config.scheduler.policy == "backfill"


def test_paths_expand_the_tilde(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, '[paths]\ndatabase = "~/custom/dispatch.db"\n'))
    assert config.paths.database == Path.home() / "custom/dispatch.db"
    assert "~" not in str(config.paths.database)


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown configuration section"):
        load_config(write(tmp_path, "[schedular]\npolicy = 'fifo'\n"))


def test_unknown_key_is_rejected_and_lists_the_valid_ones(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, "[scheduler]\nreserved_core = 4\n"))
    message = str(excinfo.value)
    assert "reserved_core" in message
    assert "reserved_cores" in message


def test_malformed_toml_names_the_file(tmp_path: Path) -> None:
    path = write(tmp_path, "[scheduler\npolicy = 'fifo'\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_wrong_type_for_a_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="expected a path string"):
        load_config(write(tmp_path, "[paths]\ndatabase = 42\n"))


def test_boolean_where_a_number_belongs_is_rejected(tmp_path: Path) -> None:
    """bool is an int subclass, so `reserved_cores = true` would otherwise mean 1."""
    with pytest.raises(ConfigError, match="expected a number"):
        load_config(write(tmp_path, "[scheduler]\nreserved_cores = true\n"))


def test_number_where_a_boolean_belongs_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="expected true or false"):
        load_config(write(tmp_path, "[daemon]\nautostart = 1\n"))


def test_a_bare_string_is_accepted_where_a_list_is_expected(tmp_path: Path) -> None:
    """A harmless slip that should not need a diagnostic."""
    config = load_config(write(tmp_path, '[notifications]\nsinks = "log"\n'))
    assert config.notifications.sinks == ("log",)


def test_lists_become_tuples(tmp_path: Path) -> None:
    config = load_config(
        write(tmp_path, '[notifications]\non = ["job.completed", "job.failed"]\n')
    )
    assert config.notifications.on == ("job.completed", "job.failed")


def test_negative_reserved_cores_is_rejected() -> None:
    with pytest.raises(ConfigError, match="reserved_cores"):
        SchedulerConfig(reserved_cores=-1)


def test_zero_heartbeat_is_rejected() -> None:
    with pytest.raises(ConfigError, match="heartbeat_s"):
        SchedulerConfig(heartbeat_s=0)


def test_total_cores_defaults_to_the_machine() -> None:
    assert SchedulerConfig().resolve_total_cores() == (os.cpu_count() or 1)


def test_total_cores_can_be_overridden() -> None:
    assert SchedulerConfig(total_cores=8).resolve_total_cores() == 8


def test_adapter_sections_pass_through_unvalidated(tmp_path: Path) -> None:
    """This layer must not know which adapters exist; each validates its own section."""
    config = load_config(
        write(tmp_path, '[adapters.openfoam]\nbashrc = "/opt/openfoam2312/etc/bashrc"\n')
    )
    assert config.adapters["openfoam"]["bashrc"] == "/opt/openfoam2312/etc/bashrc"


def test_paths_derive_from_the_runtime_directory() -> None:
    paths = Config().paths
    assert paths.socket.name == "daemon.sock"
    assert paths.socket.parent == paths.runtime_dir
    assert paths.pidfile.parent == paths.runtime_dir


def test_job_log_directory_is_per_job() -> None:
    paths = Config().paths
    assert paths.job_dir("abc-123").name == "abc-123"
    assert paths.job_dir("abc-123").parent == paths.job_log_dir


def test_xdg_data_home_is_honoured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    from dispatch.core.config import PathsConfig

    assert PathsConfig().database.is_relative_to(tmp_path / "data")


def test_runtime_dir_avoids_xdg_runtime_dir(monkeypatch, tmp_path: Path) -> None:
    """It is cleared at logout on some systems, and the daemon must outlive logout."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "volatile"))
    from dispatch.core.config import PathsConfig

    assert not PathsConfig().runtime_dir.is_relative_to(tmp_path / "volatile")
