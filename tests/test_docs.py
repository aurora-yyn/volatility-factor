# -*- coding: utf-8 -*-
"""Consistency between documentation and code.

Checks that the script names in the READMEs exist and that each script declares the correct table number.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = sorted(p for p in ROOT.glob("0*") if p.is_dir())

#: Paper table number -> script that produces it. Numbers follow the ``\newlabel`` entries of
#: ``Main Document – LaTeX.aux``.
TABLE_OWNERS = {
    1: "02_data_description/basic_statistics.py",
    2: "02_data_description/long_memory_test.py",
    3: "03_in_sample_results/in_sample_regression.py",
    4: "04_out_of_sample/oos_r2.py",
    5: "04_out_of_sample/regime_analysis.py",
    6: "04_out_of_sample/multi_factor_combinations.py",
    7: "05_robustness/mcs_test.py",
    8: "05_robustness/mcs_test.py",   # appendix MCS table, same script as Table 7
    9: "04_out_of_sample/oos_r2.py",
    10: "05_robustness/correlation_matrix_factors.py",
    11: "06_information_content/incremental_correlation.py",
    12: "06_information_content/industry_factors.py",
    13: "06_information_content/incremental_predictability.py",
    14: "07_extensions/component_factors.py",
    15: "07_extensions/component_factors.py",
    16: "07_extensions/partial_variance_factors.py",
}


@pytest.mark.parametrize("chapter", CHAPTERS, ids=lambda p: p.name)
def test_chapter_has_readme(chapter: Path):
    assert (chapter / "README.md").exists(), f"{chapter.name} has no README.md"


@pytest.mark.parametrize("chapter", CHAPTERS, ids=lambda p: p.name)
def test_readme_script_names_match(chapter: Path):
    """Every .py mentioned in a README must exist, and every .py in the folder must be listed in its README."""
    readme = chapter / "README.md"
    if not readme.exists():
        pytest.skip("README missing; reported by the previous test")

    actual = {p.name for p in chapter.glob("*.py")}
    cited = set(re.findall(r"`([a-z0-9_]+\.py)`", readme.read_text(encoding="utf-8")))

    missing = sorted(cited - actual)
    undocumented = sorted(actual - cited)
    assert not missing, f"{chapter.name}/README.md mentions scripts that do not exist: {missing}"
    assert not undocumented, f"{chapter.name} has scripts not listed in its README: {undocumented}"


@pytest.mark.parametrize("number,script", sorted(TABLE_OWNERS.items()))
def test_table_number_claimed_by_owner(number: int, script: str):
    """The script that produces a table must declare the correct table number in its docstring."""
    path = ROOT / script
    assert path.exists(), f"{script} does not exist"
    head = path.read_text(encoding="utf-8")[:1200]
    pattern = rf"Tables?\s+{number}\b|Tables?\s+\d+\s*/\s*{number}\b|{number}\s*/\s*\d+"
    assert re.search(pattern, head), (
        f"{script} does not declare Table {number} at the top; "
        f"if the number changed, update the docstring, the output file name and TABLE_OWNERS"
    )


def test_output_table_filenames_use_paper_numbers():
    """Table numbers in file names under outputs/tables/ must match TABLE_OWNERS."""
    tables = ROOT / "outputs" / "tables"
    if not tables.exists():
        pytest.skip("outputs not generated yet")

    for csv in tables.glob("table*.csv"):
        for num in re.findall(r"table(\d+)", csv.stem):
            n = int(num)
            assert n in TABLE_OWNERS, (
                f"{csv.name} uses table number {n}, which is not in TABLE_OWNERS: "
                f"either the number is wrong or the table is not registered"
            )
