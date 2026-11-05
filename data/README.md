# Data

## corpus/
`fy2025/` — CMS ICD-10-CM FY2025 (April 1, 2025 update) code files:
tabular list, alphabetic index, external-cause index, drug and neoplasm tables (XML + PDF),
plus the FY2025 coding guidelines PDF.
Source: https://www.cms.gov/files/zip/2025-code-tables-tabular-and-index-april.zip
Only the XML/XSD files are committed; the PDFs (reference only, ~90 MB) are not. Download the zip above for them.

## eval/
- `Test_Project_ICD10_Dataset.csv` — 33 de-identified encounters with transcript and
  reference ICD-10-CM codes (raw source).
  Columns: encounter_id, age, age_unit, sex, visit_reason, reference_answer, transcript.
- `eval_clean.jsonl` — cleaned encounters, one per line, with tune/heldout `split`,
  parsed `reference_codes`, and `reference_descriptions`.
- `eval_labels_long.csv` — one row per reference code.
  Columns: encounter_id, split, code, description.
