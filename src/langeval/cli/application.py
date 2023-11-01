from typing import Optional

from langeval.cli.terminal import Terminal


class Application(Terminal):
    def __init__(self, exit_func, verbosity: int, color: Optional[bool], config_file: str):
        super().__init__(verbosity, color, False)
        self.__exit_func = exit_func

        self.config_file = config_file
        self.verbosity = self.verbosity > 0

    def abort(self, text="", code=1, **kwargs):
        if text:
            self.display_error(text, **kwargs)
        self.__exit_func(code)
