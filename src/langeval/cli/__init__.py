import logging
import os
from typing import Optional

import click

from langeval.__about__ import __version__
from langeval.cli.application import Application
from langeval.cli.rerun import rerun
from langeval.cli.run import run
from langeval.cli.show import show
from langeval.config import AppEnvVars, ConfigEnvVars


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 120}, invoke_without_command=True
)
@click.option(
    "--verbose",
    "-v",
    envvar=AppEnvVars.VERBOSE,
    count=True,
    help=f"Increase verbosity (can be used additively) [env var: `{AppEnvVars.VERBOSE}`]",
)
@click.option(
    "--color/--no-color",
    default=None,
    help="Whether or not to display colored output (default is auto-detection) "
    f"[env vars: `{AppEnvVars.FORCE_COLOR}`/`{AppEnvVars.NO_COLOR}`]",
)
@click.option(
    "--config",
    "config_file",
    envvar=ConfigEnvVars.CONFIG,
    help=f"The path to a custom config file to use [env var: `{ConfigEnvVars.CONFIG}`]",
)
@click.version_option(version=__version__, prog_name="langeval")
@click.pass_context
def langeval(ctx: click.Context, verbose: int, color: Optional[bool], config_file: str):
    """
    \b
    ▄▄▌   ▄▄▄·  ▐ ▄  ▄▄ • ▄▄▄ . ▌ ▐· ▄▄▄· ▄▄▌
    ██•  ▐█ ▀█ •█▌▐█▐█ ▀ ▪▀▄.▀·▪█·█▌▐█ ▀█ ██•
    ██▪  ▄█▀▀█ ▐█▐▐▌▄█ ▀█▄▐▀▀▪▄▐█▐█•▄█▀▀█ ██▪
    ▐█▌▐▌▐█ ▪▐▌██▐█▌▐█▄▪▐█▐█▄▄▌ ███ ▐█ ▪▐▌▐█▌▐▌
    .▀▀▀  ▀  ▀ ▀▀ █▪·▀▀▀▀  ▀▀▀ . ▀   ▀  ▀ .▀▀▀
    """
    if color is None:
        if os.environ.get(AppEnvVars.NO_COLOR) == "1":
            color = False
        elif os.environ.get(AppEnvVars.FORCE_COLOR) == "1":
            color = True

    if verbose > 0:
        if verbose == 1:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.DEBUG)

    app = Application(ctx.exit, verbose, color, config_file)

    if not ctx.invoked_subcommand:
        app.display_info(ctx.get_help())
        return

    # Persist app data for sub-commands
    ctx.obj = app


langeval.add_command(run)
langeval.add_command(show)
langeval.add_command(rerun)


def main():  # no cov
    try:
        return langeval(prog_name="langeval", windows_expand_args=False) # type: ignore
    except Exception:
        from rich.console import Console

        console = Console()
        console.print_exception(suppress=[click])
        return 1
