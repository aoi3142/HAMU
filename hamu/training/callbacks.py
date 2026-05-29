from transformers import TrainerCallback
import transformers.trainer_callback
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import IntervalStrategy
import torch
from typing import Any, Optional
from accelerate.utils import gather
from accelerate import Accelerator


class FractionalLoggingCallback(TrainerCallback):
    """
    Callback to log at the next nearest interval per epoch rather than every N steps.
    Default logging in HuggingFace Trainer logs every int N steps, which can lead to logging step
    diverging from the desired interval as the number of steps per epoch may not be divisible by N.
    This callback adjusts logging to occur at the nearest interval per epoch.
    Additionally, also adds optional evaluation at specified intervals.
    """
    def __init__(self, logging_interval: Optional[float] = None, eval_interval: Optional[float] = None, longer_eval_interval: Optional[float] = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.logging_interval = logging_interval
        self.longer_eval_interval = longer_eval_interval
        self.eval_interval = eval_interval
        print(
            "FractionalLoggingCallback initialized with "
            f"logging_interval={logging_interval}, "
            f"longer_eval_interval={longer_eval_interval}, "
            f"eval_interval={eval_interval}"
        )
    def on_step_end(
        self,
        args: TrainingArguments,
        state: transformers.trainer_callback.TrainerState,
        control: transformers.trainer_callback.TrainerControl,
        **kwargs: Any,
    ) -> transformers.trainer_callback.TrainerControl:
        step = state.global_step
        steps_per_epoch = state.max_steps // state.num_train_epochs
        if steps_per_epoch <= 0:
            return control
        if self.logging_interval is not None and args.logging_strategy == IntervalStrategy.STEPS:
            if self.logging_interval < 1:
                logging_steps = self.logging_interval * steps_per_epoch
            else:
                logging_steps = self.logging_interval
            if control.should_training_stop:
                pass
            elif step > 0 and int((step + 1e-4) / logging_steps) > int((step - 1 + 1e-4) / logging_steps):
                # Log at the nearest interval
                control.should_log = True
            elif step % steps_per_epoch == 0:
                # Ensure logging at the end of each epoch
                control.should_log = True
            else:
                control.should_log = False
        if self.eval_interval is not None and self.eval_interval > 0:
            if self.longer_eval_interval is not None:
                if step <= steps_per_epoch:
                    eval_interval = self.eval_interval
                else:
                    eval_interval = self.longer_eval_interval
            else:
                eval_interval = self.eval_interval
            if eval_interval < 1:
                eval_steps = eval_interval * steps_per_epoch
            else:
                eval_steps = eval_interval
            if control.should_training_stop:
                pass
            elif step > 0 and int((step + 1e-4) / eval_steps) > int((step - 1 + 1e-4) / eval_steps):
                # Evaluate at the nearest interval
                control.should_evaluate = True
            # elif step % steps_per_epoch == 0:
            #     # Ensure evaluation at the end of each epoch
            #     control.should_evaluate = True
            elif getattr(args, "eval_strategy", None) == IntervalStrategy.STEPS:
                control.should_evaluate = False
            else:
                pass
        if control.should_evaluate:
            control.should_log = True
        return control


class NaNStoppingCallback(TrainerCallback):
    """Callback to stop training when NaN or Inf is detected in the loss, gradients, or split losses."""

    def __init__(self, accelerator: Optional[Accelerator] = None) -> None:
        super().__init__()
        self.accelerator = accelerator

    def any_device_nan_inf(self, val: Optional[float], mag: Optional[float] = None) -> bool:
        # I have absolutely no idea why I need to gather here.
        # For some reason, the trainer control doesn't propagate across devices
        # and when some device fails the nan/inf check and sets control.should_training_stop = True,
        # others don't know about it and the trainer just hangs forever
        # (some processes exits training loop but others try to keep going but wait for the dead processes forever)
        # So we gather the value across devices to make sure all devices see the same thing.
        # This causes some overhead by transferring data across devices.
        if val is None:
            return False
        val_tensor = torch.tensor([val])
        if self.accelerator is not None:
            val_tensor = val_tensor.to(device=self.accelerator.device)
            val_tensor = gather(val_tensor)
        nan_inf = torch.isnan(val_tensor).any() or torch.isinf(val_tensor).any()
        if nan_inf:
            return True
        if mag is not None and torch.abs(val_tensor).max() > mag:
            return True
        return False
    
    def on_log(self, args: Any, state: Any, control: transformers.trainer_callback.TrainerControl, logs: Optional[dict[str, Any]] = None, **kwargs: Any) -> transformers.trainer_callback.TrainerControl:
        if logs is None:
            return control

        # Check main loss
        loss = logs.get("loss")
        if self.any_device_nan_inf(loss):
            print(f"Stopping training due to NaN/Inf in loss: {loss}")
            control.should_training_stop = True
            return control
        
        # Check gradient norm
        grad_norm = logs.get("grad_norm")
        if self.any_device_nan_inf(grad_norm):
            print(f"Stopping training due to NaN/Inf in grad_norm: {grad_norm}")
            control.should_training_stop = True
            return control
        
        # Check retain loss
        retain_loss = logs.get("retain_loss")
        if self.any_device_nan_inf(retain_loss):
            print(f"Stopping training due to NaN/Inf in retain_loss: {retain_loss}")
            control.should_training_stop = True
            return control
        
        # Check forget loss
        forget_loss = logs.get("forget_loss")
        if self.any_device_nan_inf(forget_loss):
            print(f"Stopping training due to NaN/Inf in forget_loss: {forget_loss}")
            control.should_training_stop = True
            return control
        
        return control


class RetainForgetAccuracyCallback(TrainerCallback):
    """Add the CV checkpoint-selection metric used in the paper."""

    metric_name = "eval_retain_forget_accuracy"

    def on_evaluate(self, args: Any, state: Any, control: transformers.trainer_callback.TrainerControl, metrics: Optional[dict[str, Any]] = None, **kwargs: Any) -> transformers.trainer_callback.TrainerControl:
        if metrics is None:
            return control
        retain_accuracy = metrics.get("eval_retain_accuracy")
        forget_accuracy = metrics.get("eval_forget_accuracy")
        if retain_accuracy is None or forget_accuracy is None:
            return control
        metrics[self.metric_name] = (retain_accuracy + forget_accuracy) / 2.0
        return control


MyLoggingCallback = FractionalLoggingCallback
