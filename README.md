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

TODOs:

- Refactor RAG Eval
FileNotFoundError: [Errno 2] No such file or directory:
'output/ac17315_sqlcoder7b2_V4-fix/eval/eval_100_fix.jsonl'

Progress: run=Progress(total=100, finished=100, failed=0) evals={'sqleval': Progress(total=100, finished=85, failed=0)}
[2023-12-13T03:52:04.850620][task-2312131151-8f4e][runner._run] task eval sqleval progress {progress.evals[evaluator.name]}, result: {result}

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
