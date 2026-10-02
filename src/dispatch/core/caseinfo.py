"""A human-readable description of a case, as data.

``i`` on a case opens a page describing it: what solver it is, how long it will run, how
many cells its mesh has, which patches it has, what it writes, whether it is decomposed.
All of that lives in the solver's own dictionaries, and reading them is solver knowledge --
so the adapter produces this and the interface only lays it out.

That split is the whole reason this module exists. The obvious shape would have been for the
case-info screen to open ``system/controlDict`` itself, and it would have worked, and the
second solver to want the feature would have had to reimplement the screen. Here the
adapter answers a question about *its* case in terms the interface already knows how to
render, and the screen never learns what a ``controlDict`` is.

Deliberately not a mapping. A dictionary of strings would lose the ordering, the grouping
and the difference between "this case has no end time" and "I did not look", all of which
are exactly what makes the page readable rather than a dump.

See ``docs/ARCHITECTURE.md`` §9.8.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["CaseReport", "InfoField", "InfoSection", "ReportBuilder"]


@dataclass(frozen=True, slots=True)
class InfoField:
    """One labelled value.

    Attributes:
        label: What to call it, in the solver's own vocabulary -- ``End time``, ``Cells``.
            The interface does not translate these; the adapter is the one that knows what
            the number is.
        value: Already rendered. An adapter formatting its own numbers keeps units and
            precision with the knowledge of what they mean -- ``1e-05 s`` rather than a
            float the screen has to guess at.
        note: Secondary detail shown faintly beside the value, e.g. where it was read from.
        important: Draw the eye. For the handful of fields that answer "is this case what I
            think it is": the application, the mesh size, the end time.
    """

    label: str
    value: str
    note: str = ""
    important: bool = False


@dataclass(frozen=True, slots=True)
class InfoSection:
    """A titled group of fields.

    Attributes:
        title: Section heading, lowercase by the interface's convention.
        fields: Its fields, in the order they should be read.
        missing: Why the section is empty, when it is. A section that could not be read
            says so instead of vanishing: "no mesh has been generated yet" is information,
            and an absent heading is not.
    """

    title: str
    fields: Sequence[InfoField] = field(default_factory=tuple)
    missing: str = ""

    def __bool__(self) -> bool:
        return bool(self.fields) or bool(self.missing)


@dataclass(frozen=True, slots=True)
class CaseReport:
    """Everything worth knowing about one case, grouped and ordered for reading.

    Attributes:
        title: What to call the case. Usually its directory name.
        solver: The adapter that produced this.
        sections: Groups of fields, in reading order.
        warnings: Things that are true and probably unwanted -- a case half set up, a
            decomposition that does not match the mesh. Shown prominently, because the
            reason somebody opens this page is often that something is wrong.
    """

    title: str
    solver: str
    sections: Sequence[InfoSection] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return any(bool(section) for section in self.sections)

    @property
    def present(self) -> Sequence[InfoSection]:
        """Sections with something to show, in order."""
        return tuple(section for section in self.sections if section)


class ReportBuilder:
    """Accumulates a :class:`CaseReport` without the adapter juggling tuples.

    Mirrors :class:`~dispatch.core.validation.ReportBuilder`, which adapters already use for
    validation findings, so writing a ``describe_case`` feels like writing a ``validate``.
    """

    __slots__ = ("_fields", "_sections", "_solver", "_title", "_warnings")

    def __init__(self, title: str, solver: str) -> None:
        self._title = title
        self._solver = solver
        self._sections: list[InfoSection] = []
        self._fields: list[InfoField] = []
        self._warnings: list[str] = []

    def field(
        self, label: str, value: object, *, note: str = "", important: bool = False
    ) -> None:
        """Add a field to the open section, skipping it when there is nothing to say.

        ``None`` is dropped rather than rendered as "None": a case with no declared end
        time has no end time, and a row reading ``End time  None`` is worse than no row.
        An empty string is dropped for the same reason.
        """
        if value is None or value == "":
            return
        self._fields.append(
            InfoField(label=label, value=str(value), note=note, important=important)
        )

    def section(self, title: str, *, missing: str = "") -> None:
        """Close the open section and begin a new one."""
        self._close()
        self._sections.append(InfoSection(title=title, missing=missing))

    def warn(self, message: str) -> None:
        """Note something true and probably unwanted."""
        self._warnings.append(message)

    def _close(self) -> None:
        if self._fields and self._sections:
            last = self._sections[-1]
            self._sections[-1] = InfoSection(
                title=last.title, fields=tuple(self._fields), missing=last.missing
            )
        self._fields = []

    def build(self) -> CaseReport:
        """Freeze the accumulated sections into a report."""
        self._close()
        return CaseReport(
            title=self._title,
            solver=self._solver,
            sections=tuple(self._sections),
            warnings=tuple(self._warnings),
        )
