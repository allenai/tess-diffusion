from transformers import Seq2SeqTrainer
import shutil
import os
from transformers.utils import logging

logger = logging.get_logger(__name__)

class BaselineSeq2SeqTrainer(Seq2SeqTrainer):
    def annotated_split(self, split):
        return f"{split}_top_p_{self.args.top_p}_temperature_{self.args.temperature}_seed_{self.args.seed}"

    def save_metrics(self, split, metrics, combined=True):
        super().save_metrics(self.annotated_split(split), metrics, combined)

    def log_metrics(self, split, metrics):
        super().log_metrics(self.annotated_split(split), metrics)

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint)
