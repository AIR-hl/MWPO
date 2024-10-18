import warnings
from contextlib import nullcontext
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.integrations import deepspeed_init
from transformers.trainer_pt_utils import find_batch_size, EvalLoopContainer
from trl import DPOTrainer

from mwpo_config import MWPOConfig


class MWPOTrainer(DPOTrainer):
    def __init__(self, args: Optional[MWPOConfig] = None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.len_norm = args.len_norm
        if self.len_norm and args.len_scale is None:
            warnings.warn(
                "`len_scale` is not set in the MWPOConfig's init. It will default to `1.0` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.len_scale = 1.0
        self.len_scale = args.len_scale

        if args.pref_lambda is None:
            warnings.warn(
                "`pref_lambda` is not set in the MWPOConfig's init. It will default to `0.05` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.pref_lambda = 1
        self.pref_lambda = args.pref_lambda
        self.weighted = args.weighted
        if args.weight_alpha is None and self.weighted is True:
            warnings.warn(
                "`weighted_alpha` is not set in the MWPOConfig's init. It will default to `0.6` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.weighted_alpha = 0.6
        self.weight_alpha = args.weight_alpha
        if args.weight_beta is None and self.weighted is True:
            warnings.warn(
                "`weighted_alpha` is not set in the MWPOConfig's init. It will default to `0.8` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.weight_beta = 0
        self.weight_beta = args.weight_beta

        if args.len_lambda is None and self.weighted is True:
            warnings.warn(
                "`len_lambda` is not set in the MWPOConfig's init. It will default to `0.05` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.len_lambda = 0.01
        self.len_lambda = args.len_lambda

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                        _,
                        _
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _,
                    _,
                    _
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

                torch.cuda.empty_cache()
                self.accelerator.free_memory()
            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def dpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the dpo loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """

        pi_logratios = policy_chosen_logps - policy_rejected_logps

        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)

        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                    -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "robust":
            losses = (
                             -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                             + F.logsigmoid(-self.beta * logits) * self.label_smoothing
                     ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        elif self.loss_type == "bco_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean

            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        elif self.loss_type == "sppo_hard":
            # In the paper (https://arxiv.org/pdf/2405.00675), SPPO employs a soft probability approach, estimated using the PairRM score. The probability calculation is conducted outside of the trainer class. The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta
            losses = (
                    -F.logsigmoid(chosen_rewards)
                    - 0.5 * F.logsigmoid(-chosen_rewards)
                    - 0.5 * F.logsigmoid(-rejected_rewards)
            )
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exp', 'truncated', 'savage']"
            )

        chosen_rewards = (
                self.beta
                * (
                        policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
                ).detach()
        )
        rejected_rewards = (
                self.beta
                * (
                        policy_rejected_logps.to(self.accelerator.device)
                        - reference_rejected_logps.to(self.accelerator.device)
                ).detach()
        )

        return losses, chosen_rewards, rejected_rewards, logits.detach()

    def concatenated_forward(
            self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        if self.is_encoder_decoder:
            labels = concatenated_batch["concatenated_labels"].clone()
        else:
            labels = concatenated_batch["concatenated_input_ids"].clone()
            attention_mask = concatenated_batch["concatenated_attention_mask"]
            labels = torch.where(attention_mask == 1, labels, self.label_pad_token_id)

        if self.args.rpo_alpha != 0 or self.args.rpo_alpha is not None:
            nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])
        else:
            nll_loss = torch.tensor(0.0).to(self.accelerator.device)
        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        if self.loss_type == "ipo":
            all_logps = all_logps / size_completion

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_size = size_completion[:len_chosen]
        rejected_size = size_completion[len_chosen:]
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, chosen_size, rejected_size)

    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
            chosen_size,
            rejected_size
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
                "reference_chosen_logps" in batch
                and "reference_rejected_logps" in batch
                and self.args.rpo_alpha is not None
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                            _,
                            _
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                        _,
                        _
                    ) = self.concatenated_forward(self.ref_model, batch)

        avg_chosen_logp = policy_chosen_logps.detach() / chosen_size
        avg_rejected_logp = policy_rejected_logps.detach() / rejected_size

        losses, chosen_rewards, rejected_rewards, logits = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}avg_logps/margins"] = (avg_chosen_logp - avg_rejected_logp).mean().cpu()
        metrics[f"{prefix}avg_logps/chosen"] = avg_chosen_logp.mean().cpu()
        metrics[f"{prefix}avg_logps/rejected"] = avg_rejected_logp.mean().cpu()
        if self.weighted:
            if self.len_norm:
                weight1 = F.sigmoid(self.len_scale * (chosen_rewards / chosen_size - rejected_rewards / rejected_size)) ** self.weight_alpha
            else:
                weight1 = F.sigmoid(0.5 * (chosen_rewards - rejected_rewards)) ** self.weight_alpha

            weight2 = F.sigmoid(self.len_lambda * (rejected_size - chosen_size)) ** self.weight_beta

            losses = losses * weight1 * weight2

        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}pref_loss"] = losses.detach().mean().cpu()
            metrics[f"{prefix}sft_loss"] = policy_nll_loss.detach().mean().cpu()
            losses = losses + policy_nll_loss * self.args.rpo_alpha

        return losses.mean(), metrics