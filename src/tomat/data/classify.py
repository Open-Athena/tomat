"""Chemical-category classification matching Yael/Betsey's sparsity table.

The doc reports voxel-sparsity statistics broken out by material category
(halide, oxyhalide, chalcogenide, oxychalcogenide, oxide, intermetallic,
other). Those categories are derivable from the elements present in each
structure — no Materials Project API calls required.

Rules (applied top-down; first match wins):

1. **intermetallic** — every element is a metal (no nonmetals at all).
2. **halide** — contains at least one halogen (F, Cl, Br, I), no oxygen,
   no chalcogen from {S, Se, Te}.
3. **oxyhalide** — contains a halogen and oxygen (chalcogens may also
   be present; halogen+O takes precedence over chalcogen).
4. **chalcogenide** — contains S/Se/Te, no oxygen, no halogen.
5. **oxychalcogenide** — contains S/Se/Te and oxygen, no halogen.
6. **oxide** — contains oxygen, no halogen, no chalcogen from {S, Se, Te}.
7. **other** — anything else (e.g. nitrides, carbides, borides, hydrides,
   nonmetal-only).
"""

from typing import Iterable

from pymatgen.core.periodic_table import Element

HALOGENS = frozenset({"F", "Cl", "Br", "I"})
CHALCOGENS_NON_O = frozenset({"S", "Se", "Te"})

CATEGORIES = (
    "halide",
    "oxyhalide",
    "oxychalcogenide",
    "oxide",
    "chalcogenide",
    "other",
    "intermetallic",
)


def classify_elements(symbols: Iterable[str]) -> str:
    elements = set(symbols)
    has_halogen = bool(elements & HALOGENS)
    has_oxygen = "O" in elements
    has_chalcogen = bool(elements & CHALCOGENS_NON_O)
    all_metals = all(Element(sym).is_metal for sym in elements)

    if all_metals:
        return "intermetallic"
    if has_halogen and has_oxygen:
        return "oxyhalide"
    if has_halogen:
        return "halide"
    if has_chalcogen and has_oxygen:
        return "oxychalcogenide"
    if has_chalcogen:
        return "chalcogenide"
    if has_oxygen:
        return "oxide"
    return "other"
