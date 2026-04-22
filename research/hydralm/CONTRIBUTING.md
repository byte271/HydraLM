# Contributing to HydraLM

Thanks for taking the time to contribute. HydraLM is a small research
codebase, so the bar for contributions is simple: the change has to be
correct, easy to read, and covered by tests.

## Ground rules

- Be respectful. This project follows the
  [Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md).
- Keep pull requests focused. One logical change per PR.
- Every behavioural change needs a test (or a very good reason not to).
- Public APIs are the ones re-exported from `hydralm/__init__.py` and its
  sub-package `__init__.py` files. Changes there need a CHANGELOG entry.

## Getting set up

HydraLM targets Python 3.10+ and PyTorch 2.2+.

```bash
git clone https://github.com/byte271/hydralm.git
cd hydralm
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extra pulls in `pytest`, `ruff`, and `mypy`. CUDA is optional;
the whole test suite runs on CPU.

## Running the checks

```bash
pytest                               # full test suite
pytest tests/test_shapes.py -k gdn   # single test
ruff check .                         # lint
ruff format --check .                # formatting
mypy hydralm                         # type-check
```

Everything above is also run in CI on every pull request.

## Filing issues

Before opening an issue, please:

1. Search existing issues at
   https://github.com/byte271/hydralm/issues to avoid duplicates.
2. Include the exact HydraLM version
   (`python -c "import hydralm; print(hydralm.__version__)"`),
   the PyTorch version, CUDA version if applicable, and the minimal
   snippet needed to reproduce.
3. For performance issues, include the shapes, dtype, and device.

Security-sensitive reports should go through the process described in
[SECURITY.md](./SECURITY.md) instead of the public tracker.

## Proposing changes

1. Fork the repository and create a feature branch off `main`.
2. Make the change. Keep the diff small and readable.
3. Add or update tests in `tests/`.
4. Update `docs/` and `CHANGELOG.md` if you changed anything user-facing.
5. Open a PR against `byte271/hydralm:main` and fill in the template.
6. CI must be green before review.

### Style

- Follow the style enforced by `ruff format`. In short: 4-space indent,
  120-column soft limit, double quotes, trailing commas in multi-line
  literals.
- Docstrings are Google-style. Public functions and classes must have
  one.
- Prefer explicit keyword arguments at call sites for anything other
  than `(tensor)` / `(tensor, mask)`.
- Type-annotate public signatures. Internals are optional but
  encouraged.

### Tests

- Put CPU-only tests in `tests/`.
- Keep unit tests fast (< 1 s each). Slow end-to-end or GPU tests go
  under an `@pytest.mark.slow` marker and are skipped by default.
- When adding a new module, add a shape test and, where applicable, an
  equivalence test (recurrent form vs. parallel form).

### Commits

We squash-merge, so the PR title becomes the commit message. Use the
[Conventional Commits](https://www.conventionalcommits.org/) prefix
(`feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `chore:`) for
the PR title.

## Releasing

Maintainers cut releases by:

1. Bumping `version` in `pyproject.toml` and `hydralm/__init__.py`.
2. Moving the `[Unreleased]` section of `CHANGELOG.md` to a new
   version heading.
3. Tagging `vX.Y.Z` on `main` and pushing the tag.
4. GitHub Actions builds the sdist/wheel and publishes to PyPI.

## License

By contributing, you agree that your contributions will be licensed
under the [MIT License](./LICENSE).
