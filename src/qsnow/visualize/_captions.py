"""The chip-level caption format shared by the matplotlib figure modules.

`profiling.py` (sweep outcomes) and `distributions.py` (sweep inputs) both title their
figures with a headline, the chip's noise model line, and an optional caller suffix.
Keeping the format here means a change to how a chip is described lands on every
figure at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from qsnow.interface.chip import Chip


def noise_caption(chip: "Chip", shots: Optional[int]) -> str:
    """The `PED(type=..., mean=..., dev=..., shots=...)` line under a figure title.

    `shots` is omitted when None: a chip-only figure has no sampling to report.
    """
    noise_model = chip.summary().get("noise_model") or {}
    name = noise_model.get("name", "custom")
    p = np.array([n.p for n in chip.noise_map.values()], dtype=float)
    fields = [f"type={name}", f"mean={p.mean():.2g}", f"dev={p.std():.2g}"]
    if shots is not None:
        fields.append(f"shots={shots}")
    return f"PED({', '.join(fields)})"


def chip_headline(chip: "Chip", subject: str) -> str:
    """`<subject> across LxH chip`, with the size in unit cells rather than coordinates."""
    length, height = chip.unit_dims
    return f"{subject} across {length}x{height} chip"


def chip_suptitle(
    headline: str, chip: "Chip", shots: Optional[int], add_title: str = ""
) -> str:
    """Headline, noise caption, and caller suffix, one per line, blanks dropped."""
    return "\n".join(
        line for line in (headline, noise_caption(chip, shots), add_title) if line
    )
