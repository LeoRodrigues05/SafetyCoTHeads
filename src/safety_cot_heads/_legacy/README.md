"""Verbatim ports of upstream source code.

This package preserves the original files from the two upstream repositories
so they are part of the codebase and can be cited / re-used directly:

* ``_legacy/sha/`` — files lifted verbatim from
  https://github.com/ydyjya/SafetyHeadAttribution (``lib/`` subtree, the
  rule-based discriminator, and source-bearing notebooks).
* ``_legacy/cots/`` — files lifted verbatim from
  https://github.com/Lott11/CoT-safety .

These files are kept *as-is* (apart from the small in-file patch notes added
where a hardcoded path / device caused breakage at import time).  The clean,
config-driven public API lives in the sibling packages
(:mod:`safety_cot_heads.models`, :mod:`safety_cot_heads.attribution`, etc.)
and re-uses these legacy implementations where it makes sense.

See ``docs/general/MIGRATION.md`` and ``docs/general/PREVIOUS_CODE_MAP.md`` at
the workspace root for the per-file mapping.
"""
