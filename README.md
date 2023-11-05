# LangEval

Evaluation for AI apps and agent

[![PyPI - Version](https://img.shields.io/pypi/v/langeval-cli.svg)](https://pypi.org/project/langeval-cli)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/langeval-cli.svg)](https://pypi.org/project/langeval-cli)

```txt
▄▄▌   ▄▄▄·  ▐ ▄  ▄▄ • ▄▄▄ . ▌ ▐· ▄▄▄· ▄▄▌
██•  ▐█ ▀█ •█▌▐█▐█ ▀ ▪▀▄.▀·▪█·█▌▐█ ▀█ ██•
██▪  ▄█▀▀█ ▐█▐▐▌▄█ ▀█▄▐▀▀▪▄▐█▐█•▄█▀▀█ ██▪
▐█▌▐▌▐█ ▪▐▌██▐█▌▐█▄▪▐█▐█▄▄▌ ███ ▐█ ▪▐▌▐█▌▐▌
.▀▀▀  ▀  ▀ ▀▀ █▪·▀▀▀▀  ▀▀▀ . ▀   ▀  ▀ .▀▀▀
```

-----

## Table of Contents

- [Installation](#installation)
- [Documents](#documents)
- [How to use](#how-to-use)
- [Development](#development)
- [License](#license)

## Installation

```console
pip install langeval-cli
```

## Documents

TODO

## How to use

see `./examples` for more details.

## Development

```bash
# Create virtual environment
hatch env create
# Activate virtual environment
hatch shell
# Run test
hatch run test
# Run lint
hatch run lint:style

# Version dump
hatch version patch/minor/major
# Build
hatch build
# Upload to pypi
hatch publish
```

## License

`LangEval` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## TODO

| Priority | Description |
| -------- | ----------- |
| High     |    Re-run failed task.         |
| High     | Better Custom Provider. |
| Medium | Documents |
| Medium | Test |
| Medium | Pass the dataset |
| Low | Support multi evaluator in one task. |
| Low | Support run task from Python code. |
| Low | Display task status in web. |
| Low     |    Support OpenAI function calling. |