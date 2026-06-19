import math

from abc import abstractmethod
from functools import partial
from typing import Tuple
from typing import Union

import torch

from torch.nn import functional as F


class LigerFusedLinearKDBase(torch.autograd.Function):
    @abstractmethod
    def distillation_loss_fn(
        student_logits,
        teacher_logits,
        target,
        ignore_index=-100,
        fp32_upcast=True,
        temperature=1.0,
        kd_token_mask=None,
    ):
        """
        Compute a soft KD loss term on logits for a single chunk.

        Args:
            student_logits: Student logits for tokens in the chunk.
            teacher_logits: Teacher logits for the same tokens.
            target: Token labels used only for `ignore_index` masking.
            ignore_index: Label value excluded from loss accumulation.
            fp32_upcast: Whether to compute softmax/log-softmax in FP32.
            temperature: Logit temperature used to soften both distributions.

        Returns:
            torch.Tensor: Summed KD loss over valid tokens in the chunk.
        """
        # (B, S, V) -> (BxS, V)
        student_logits = student_logits.view(-1, student_logits.shape[-1])
        teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])
        target = target.view(-1)

        if fp32_upcast:
            student_logits = student_logits.float()
            teacher_logits = teacher_logits.float()

        if temperature != 1.0:
            student_logits = student_logits.mul(1.0 / temperature)
            teacher_logits = teacher_logits.mul(1.0 / temperature)
        
        student_logprobs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        # mask out infinities originating *only* from student logits
        # (teacher logits infs are extremely rare and do not
        # affect gradients w.r.t. student parameters).
        inf_mask = torch.isinf(student_logits)
        kd_loss_per_token = - torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0).sum(-1)
        
        # T² scaling: dividing logits by T scales gradients by 1/T², so we multiply the loss by
        # T² to keep gradient magnitudes independent of temperature (Hinton et al., 2015).
        if temperature != 1.0:
            kd_loss_per_token = kd_loss_per_token * (temperature**2)
        
        # Apply ignore_index mask
        mask = target != ignore_index
        if kd_token_mask is not None:
            mask = mask & (kd_token_mask.reshape(-1) > 0)
        kd_loss_per_token = kd_loss_per_token.masked_fill(~mask, 0.0)

        return kd_loss_per_token.sum()
        
    
    @staticmethod
    def chunk_forward(
        student_input_chunk,
        student_weight,
        teacher_input_chunk,
        teacher_weight,
        target_chunk,
        student_bias=None,
        teacher_bias=None,
        ignore_index=-100,
        compute_ce_loss=True,
    ):
        # Student
        student_logits_chunk = student_input_chunk @ student_weight.t()
        if student_bias is not None:
            student_logits_chunk += student_bias

        # Teacher
        with torch.no_grad():
            teacher_logits_chunk = teacher_input_chunk @ teacher_weight.t()
            if teacher_bias is not None:
                teacher_logits_chunk += teacher_bias

        # The hard/task loss
        ce_loss = torch.tensor(0.0, device=student_logits_chunk.device, dtype=student_logits_chunk.dtype)
        if compute_ce_loss:
            student_log_probs_chunk = F.log_softmax(student_logits_chunk.float(), dim=-1)
            ce_loss = F.nll_loss(
                student_log_probs_chunk.view(-1, student_log_probs_chunk.shape[-1]),
                target_chunk.view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )

        return student_logits_chunk, teacher_logits_chunk, ce_loss
    

    @staticmethod
    def _compute_loss(
        student_input_chunk,
        student_weight,
        teacher_input_chunk,
        teacher_weight,
        target_chunk,
        student_bias=None,
        teacher_bias=None,
        ignore_index=-100,
        compute_ce_loss=True,
        num_batch_labels=None,
        weight_hard_loss=0.5,
        weight_soft_loss=0.5,
        distillation_loss_fn=None,
        **loss_kwargs,
    ):
        """
        Compute one chunk loss as a weighted sum of hard CE and soft KD losses.

        Args:
            student_input_chunk: Student hidden states for this chunk.
            student_weight: Student output projection weight.
            teacher_input_chunk: Teacher hidden states for this chunk.
            teacher_weight: Teacher output projection weight.
            target_chunk: Labels for this chunk.
            student_bias: Optional student output projection bias.
            teacher_bias: Optional teacher output projection bias.
            ignore_index: Label value excluded from hard/soft loss accumulation.
            compute_ce_loss: Whether to include hard CE loss.
            num_batch_labels: Optional normalization denominator for both losses.
            weight_hard_loss: Coefficient for hard CE loss.
            weight_soft_loss: Coefficient for soft KD loss.
            distillation_loss_fn: Chunk-level KD loss function.
            loss_kwargs: Extra keyword arguments passed into `distillation_loss_fn`.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
                `(loss, (soft_loss, hard_loss, student_logits_chunk, teacher_logits_chunk))`.
        """
        (
            student_logits_chunk,
            teacher_logits_chunk,
            hard_loss,
        ) = LigerFusedLinearKDBase.chunk_forward(
            student_input_chunk,
            student_weight,
            teacher_input_chunk,
            teacher_weight,
            target_chunk,
            student_bias=student_bias,
            teacher_bias=teacher_bias,
            ignore_index=ignore_index,
            compute_ce_loss=compute_ce_loss,
        )

        # If the teacher and student token size is different, pad student logits to match the teacher's.
        # This only applies to cases where they share exactly the same vocab and tokenizer just
        # that teacher logit is padded for some training efficiency such as
        # https://huggingface.co/Qwen/Qwen1.5-72B-Chat/discussions/1#662883f568adf59b07b176d2
        # teacher_vocab_size = teacher_weight.shape[0]
        # student_vocab_size = student_weight.shape[0]
        # if teacher_vocab_size > student_vocab_size:
            # pad_size = teacher_vocab_size - student_vocab_size
            # pad_tensor = torch.zeros(
                # (*student_logits_chunk.shape[:-1], pad_size),
                # dtype=student_logits_chunk.dtype,
                # device=student_logits_chunk.device,
            # )
            # student_logits_chunk = torch.cat([student_logits_chunk, pad_tensor], dim=-1)

        soft_loss = distillation_loss_fn(
            student_logits_chunk,
            teacher_logits_chunk,
            target=target_chunk,
            ignore_index=ignore_index,
            **loss_kwargs,
        )

        if num_batch_labels is not None:
            if num_batch_labels == 0:
                hard_loss = hard_loss * 0.0
                soft_loss = soft_loss * 0.0
            else:
                hard_loss /= num_batch_labels
                soft_loss /= num_batch_labels

        loss = weight_hard_loss * hard_loss + weight_soft_loss * soft_loss
        return loss, (soft_loss, hard_loss, student_logits_chunk, teacher_logits_chunk)

    @staticmethod
    def forward(
        cls,
        ctx,
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        target,
        student_bias=None,
        teacher_bias=None,
        ignore_index=-100,
        compute_ce_loss=True,
        num_batch_labels=None,
        weight_hard_loss=0.5,
        weight_soft_loss=0.5,
        chunk_size=1024,
        compiled=True,
        return_soft_hard_loss=False,
        **loss_kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Run fused linear KD over chunks and accumulate student-side gradients.

        Args:
            student_input: Flattened student hidden states.
            student_weight: Student output projection weight.
            teacher_input: Flattened teacher hidden states.
            teacher_weight: Teacher output projection weight.
            target: Flattened token labels.
            student_bias: Optional student output projection bias.
            teacher_bias: Optional teacher output projection bias.
            ignore_index: Label value excluded from loss accumulation.
            compute_ce_loss: Whether to compute hard CE loss.
            num_batch_labels: Optional normalization denominator for both losses.
            weight_hard_loss: Coefficient for hard CE loss.
            weight_soft_loss: Coefficient for soft KD loss.
            chunk_size: Number of flattened tokens processed per chunk.
            compiled: Whether to `torch.compile` the per-chunk accumulation function.
            return_soft_hard_loss: Whether to also return accumulated soft/hard terms.
            loss_kwargs: Extra keyword arguments forwarded to `distillation_loss_fn`.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Total loss, or `(total_loss, soft_loss, hard_loss)` when
                `return_soft_hard_loss=True`.
        """
        CHUNK_SIZE = chunk_size
        grad_weight = torch.zeros_like(student_weight)
        grad_inputs = []
        grad_bias = torch.zeros_like(student_bias) if student_bias is not None else None
        loss_acc = torch.zeros((), device=student_input.device)
        soft_loss_acc = torch.zeros((), device=student_input.device) if return_soft_hard_loss else None
        hard_loss_acc = torch.zeros((), device=student_input.device) if return_soft_hard_loss else None

        loss_func_to_call = partial(
            LigerFusedLinearKDBase._compute_loss,
            ignore_index=ignore_index,
            compute_ce_loss=compute_ce_loss,
            num_batch_labels=num_batch_labels,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            distillation_loss_fn=cls.distillation_loss_fn,
            **loss_kwargs,
        )

        def accumulate_chunk(student_input_chunk, teacher_input_chunk, target_chunk):
            if student_bias is not None:
                (
                    (chunk_grad_input, chunk_grad_weight, chunk_grad_bias),
                    (
                        chunk_loss,
                        (
                            chunk_soft_loss,
                            chunk_hard_loss,
                            chunk_student_logits,
                            chunk_teacher_logits,
                        ),
                    ),
                ) = torch.func.grad_and_value(loss_func_to_call, argnums=(0, 1, 5), has_aux=True)(
                    student_input_chunk,
                    student_weight,
                    teacher_input_chunk,
                    teacher_weight,
                    target_chunk,
                    student_bias,
                    teacher_bias,
                )
                grad_bias.add_(chunk_grad_bias)
            else:
                (
                    (chunk_grad_input, chunk_grad_weight),
                    (
                        chunk_loss,
                        (
                            chunk_soft_loss,
                            chunk_hard_loss,
                            chunk_student_logits,
                            chunk_teacher_logits,
                        ),
                    ),
                ) = torch.func.grad_and_value(loss_func_to_call, argnums=(0, 1), has_aux=True)(
                    student_input_chunk,
                    student_weight,
                    teacher_input_chunk,
                    teacher_weight,
                    target_chunk,
                    student_bias,
                    teacher_bias,
                )
            grad_weight.add_(chunk_grad_weight)
            loss_acc.add_(chunk_loss)
            if return_soft_hard_loss:
                soft_loss_acc.add_(chunk_soft_loss)
                hard_loss_acc.add_(chunk_hard_loss)
            return chunk_grad_input

        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)

        num_chunks = max(1, math.ceil(student_input.shape[0] / CHUNK_SIZE))
        _student_input_chunks = torch.chunk(student_input, chunks=num_chunks, dim=0)
        _teacher_input_chunks = torch.chunk(teacher_input, chunks=num_chunks, dim=0)
        _target_chunks = torch.chunk(target, chunks=num_chunks, dim=0)
        for student_input_chunk, teacher_input_chunk, target_chunk in zip(
            _student_input_chunks, _teacher_input_chunks, _target_chunks
        ):
            grad_input = accumulate_chunk(student_input_chunk, teacher_input_chunk, target_chunk)
            grad_inputs.append(grad_input)

        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0),
            grad_weight,
            grad_bias,
        )
        if return_soft_hard_loss:
            return loss_acc, soft_loss_acc, hard_loss_acc
        return loss_acc

    @staticmethod
    def backward(ctx, grad_output, *args):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):
            grad_input = grad_input * grad_output
            grad_weight = grad_weight * grad_output
            grad_bias = grad_bias * grad_output if grad_bias is not None else None

        return grad_input, grad_weight, None, None, None, grad_bias


class LigerFusedLinearKDFunction(LigerFusedLinearKDBase):
    @classmethod
    def forward(
        cls,
        ctx,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        target: torch.LongTensor,
        student_bias: torch.Tensor = None,
        teacher_bias: torch.Tensor = None,
        ignore_index: int = -100,
        fp32_upcast: bool = True,
        temperature: float = 1.0,
        compiled: bool = True,
        chunk_size: int = 1024,
        num_batch_labels: int = None,
    ) -> torch.Tensor:
        return super().forward(
            cls=cls,
            ctx=ctx,
            student_input=student_input,
            student_weight=student_weight,
            teacher_input=teacher_input,
            teacher_weight=teacher_weight,
            target=target,
            student_bias=student_bias,
            teacher_bias=teacher_bias,
            ignore_index=ignore_index,
            compute_ce_loss=False,
            num_batch_labels=num_batch_labels,
            weight_hard_loss=0.0,
            weight_soft_loss=1.0,
            chunk_size=chunk_size,
            compiled=compiled,
            return_soft_hard_loss=False,
            fp32_upcast=fp32_upcast,
            temperature=temperature,
        )

    @staticmethod
    def backward(ctx, grad_output, *args):
        grads = LigerFusedLinearKDBase.backward(ctx, grad_output, *args)[:6]
        return (
            *grads,
            None,  # teacher_bias
            None,  # ignore_index
            None,  # fp32_upcast
            None,  # temperature
            None,  # compiled
            None,  # chunk_size
            None,  # num_batch_labels
        )


class LigerFusedKDSoftLoss(torch.nn.Module):
    """
    Fused linear layer with soft KD only (no hard CE term).
    """

    def __init__(
        self,
        ignore_index: int = -100,
        fp32_upcast: bool = True,
        temperature: float = 1.0,
        compiled: bool = True,
        chunk_size: int = 1024,
    ):
        """
        Configure the soft-KD fused loss module.

        Args:
            ignore_index: Label value excluded from soft KD accumulation.
            fp32_upcast: Whether to compute softmax/log-softmax in FP32.
            temperature: Logit temperature for teacher and student distributions.
            compiled: Whether to `torch.compile` chunk accumulation.
            chunk_size: Number of flattened tokens processed per chunk.
        """
        super().__init__()
        assert temperature != 0.0, "Temperature cannot be 0."
        self.ignore_index = ignore_index
        self.fp32_upcast = fp32_upcast
        self.temperature = temperature
        self.compiled = compiled
        self.chunk_size = chunk_size

    def forward(
        self,
        student_hidden_states: torch.Tensor,
        teacher_hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        student_lm_weight: torch.Tensor,
        teacher_lm_weight: torch.Tensor,
        num_batch_labels: int | None = None,
    ) -> torch.Tensor:
        """
        Compute soft KD loss between teacher and student token distributions.

        Args:
            student_hidden_states: Flattened student hidden states.
            teacher_hidden_states: Flattened teacher hidden states.
            labels: Token labels used for ignore-index masking.
            student_lm_weight: Student output projection weight.
            teacher_lm_weight: Teacher output projection weight.
            num_batch_labels: Optional normalization denominator.

        Returns:
            torch.Tensor: Scalar soft KD loss.
        """
        hidden_size = student_hidden_states.shape[-1]
        student_hidden_states = student_hidden_states.view(-1, hidden_size)
        teacher_hidden_states = teacher_hidden_states.view(-1, hidden_size)
        labels = labels.view(-1)

        return LigerFusedLinearKDFunction.apply(
            student_hidden_states,
            student_lm_weight,
            teacher_hidden_states,
            teacher_lm_weight,
            labels,
            None,  # student_bias
            None,  # teacher_bias
            self.ignore_index,
            self.fp32_upcast,
            self.temperature,
            self.compiled,
            self.chunk_size,
            num_batch_labels,
        )


