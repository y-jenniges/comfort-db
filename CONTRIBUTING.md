# Contributing
Thank you for your interest in contributing to `comfort-db`! 
If you would like to contribute bug reports, new feature ideas or improve/add to the existing code, please have a look
at this document for good practices. 

For a general overview and introduction to the library, please refer to the README.md.

---

## Reporting a bug
If you find a bug, we would appreciate a short report including
- The operating system,
- The Python version,
- The `comfort-db` version,
- The used data (e.g. the COMFORT dataset version or the sample data),
- A minimal code snippet reproducing the error and
- The full traceback.

A bug-report issue template can be found [here](https://github.com/y-jenniges/comfort-db/issues/new/choose). 

## Propose a new feature
If you have new ideas, please first open an issue before writing code, so we can discuss and agree on a design.
A new-feature issue template can be found [here](https://github.com/y-jenniges/comfort-db/issues/new/choose). 

## Develop code
1. Clone the repository: `git clone https://github.com/y-jenniges/comfort-db.git`
2. Create and activate a new, empty Python environment (e.g. via `conda` or `venv`).
3. From the repository root, install the package in editable mode with all development dependencies:
   `pip install -e ".[dev]"`
4. Copy `.env.example` to `.env` and set `COMFORT_DB_PATH` to your local COMFORT database.
   The other paths are optional (see the comments in the `.env.example` file).

## Tests
Run the tests from the repository root with `pytest`. Tests marked `needs_bathymetry` are skipped unless `BATHYMETRY_PATH`
in your `.env` points to an existing GEBCO NetCDF file.

## Pull requests
Branch from main and keep your PR focused. 
Ensure to add or update the tests and the documentation to cover the branch's updates.
When you open the pull request, the description is pre-filled from the [PR template](.github/pull_request_template.md). 
Please fill it with a short summary, the changes you made and if all tests passed. 
Ensure that the CI is green.

## Docstring and comment style
Add short comments in the code to guide the reader.

Write docstrings for functions and classes in Google style. 

Sphinx will infer the data types from the function signature, so include the argument types in the signature (not in the `Args` part of the docstring). 
If this is not possible for a specific case, add the type to the docstring. 

We use British English, no Oxford commas and capitalise after colons. 


---
Thank you for adhering to this repo's practices and your support!
