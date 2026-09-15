"""Shared visual theme: white background, dark-blue accents.

A single source of truth for colors used by both the desktop GUI (Qt
stylesheet) and the HTML processing report, so the two never drift into
different palettes.
"""

from __future__ import annotations

BACKGROUND = "#FFFFFF"
SURFACE = "#F4F8FC"  # very light blue-tinted panel/table background
PRIMARY_DARK = "#0B2E4F"  # dark navy: headers, titles, primary buttons
PRIMARY = "#1D4E89"  # medium blue: interactive elements, links, plot markers
BORDER = "#C7D6E5"
TEXT = "#1A1A1A"
TEXT_MUTED = "#5A6B7B"
PENDING = "#A15C00"  # amber, reserved for "not yet available" notices
