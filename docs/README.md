# Traffic Prophet Guides

This folder contains tutorials on how to use Traffic Prophet, and how to hack it
to produce updated AADT estimates with TEPS.

- `getting_started` - tutorials on how to run Traffic Prophet.
  - `Reading Data.ipynb` - reading data from Postgres.
  - `Running CountMatch.ipynb` -  running CountMatch.
- `teps` - tutorial on how to hack Traffic Prophet to run TEPS
  - `Convert Postgres Data to Zips.ipynb` - convert Postgres data (covered in
    `getting_started/Reading Data.ipynb`) to TEPS-readable zip files.
  - `TEPs Output Analysis.ipynb` - basic sensibility and self-consistency checks
    of TEPS outputs.
  - `Running TEPS.md` - step-by-step tutorial to run TEPS-I.
