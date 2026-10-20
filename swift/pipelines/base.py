# Copyright (c) ModelScope Contributors. All rights reserved.
import datetime as dt
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import swift
from swift.arguments import BaseArguments
from swift.utils import (ProcessorMixin, get_logger, parse_args, seed_everything,
                         suppress_transformers_tied_weights_warning)

logger = get_logger()


class SwiftPipeline(ABC, ProcessorMixin):
    args_class = BaseArguments

    def __init__(self, args: Optional[Union[List[str], args_class]] = None):
        suppress_transformers_tied_weights_warning()
        self.args = self._parse_args(args)
        args = self.args
        logger.info(f'args: {args}')
        self._set_seed()

    def _set_seed(self):
        args = self.args
        if hasattr(args, 'seed'):
            seed = args.seed + max(getattr(args, 'rank', -1), 0)
            seed_everything(seed)
            logger.info(f'Global seed set to {seed}')

    def _parse_args(self, args: Optional[Union[List[str], args_class]] = None) -> args_class:
        if isinstance(args, self.args_class):
            return args
        assert self.args_class is not None
        args, remaining_argv = parse_args(self.args_class, args)
        if len(remaining_argv) > 0:
            if getattr(args, 'ignore_args_error', False):
                logger.warning(f'remaining_argv: {remaining_argv}')
            else:
                raise ValueError(f'remaining_argv: {remaining_argv}')
        return args

    def main(self):
        logger.info(f'Start time of running main: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
        logger.info(f'swift.__version__: {swift.__version__}')
        result = self.run()
        logger.info(f'End time of running main: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
        return result

    @abstractmethod
    def run(self):
        pass
