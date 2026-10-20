# Copyright (c) ModelScope Contributors. All rights reserved.
import gc
import os

from .base import MegatronCallback


class DefaultFlowCallback(MegatronCallback):

    @staticmethod
    def _is_epoch_boundary(iteration: int, train_iters: int, num_train_epochs: int) -> bool:
        """Return true exactly once for each evenly distributed epoch boundary.

        ``train_iters`` may not be divisible by ``num_train_epochs`` after the
        dataset/global-batch flooring in MegatronArguments.init_iters. Using a
        fixed floored save_steps would otherwise create an extra final
        checkpoint and can evict the first epoch checkpoint.
        """
        if iteration <= 0 or train_iters <= 0 or num_train_epochs <= 0:
            return False
        previous_epoch = (iteration - 1) * num_train_epochs // train_iters
        current_epoch = iteration * num_train_epochs // train_iters
        return current_epoch > previous_epoch

    def on_train_begin(self):
        args = self.args
        if args.manual_gc:
            gc.disable()
            gc.collect()

    def on_step_end(self):
        args = self.args
        state = self.state

        state.consumed_train_samples += args.global_batch_size

        if state.iteration == 1 or state.iteration % args.logging_steps == 0:
            state.should_log = True
        if args.eval_steps and state.iteration % args.eval_steps == 0 and args.eval_iters > 0:
            state.should_eval = True
        if args.save_strategy == 'epoch' and args.num_train_epochs:
            if self._is_epoch_boundary(state.iteration, args.train_iters, args.num_train_epochs):
                state.should_save = True
        elif args.save_steps and state.iteration % args.save_steps == 0:
            state.should_save = True

        skip_final_save = os.environ.get('MEGATRON_SKIP_FINAL_SAVE', '').lower() in {'1', 'true', 'yes', 'on'}
        if state.iteration >= args.train_iters:
            if args.eval_iters > 0:
                state.should_eval = True
            if not skip_final_save:
                state.should_save = True
        if args.manual_gc and args.manual_gc_steps != 0 and state.iteration % args.manual_gc_steps == 0:
            gc.collect()

    def on_eval_begin(self):
        args = self.args
        if args.manual_gc and args.manual_gc_eval:
            gc.collect()

    def on_eval_end(self):
        args = self.args
        if args.manual_gc and args.manual_gc_eval:
            gc.collect(generation=0)
