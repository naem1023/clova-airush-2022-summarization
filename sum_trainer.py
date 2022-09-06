from transformers import Seq2SeqTrainer, is_datasets_available, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
import torch
import torch.nn.functional as F

def RankingLoss(score, summary_score=None, margin=0.001, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(margin)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


class SummaryTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # print("coupute loss: ", outputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            # if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            #     loss = self.label_smoother(outputs, labels, shift_labels=True)
            # else:
            loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # print(outputs.logits.shape)

            labels = inputs.pop("labels")
            # output = output.view(batch_size, -1, output.size(1), output.size(2)) # [bz, cand_num, seq_len, word_dim]
            # output = output[0]  # [bz x cand_num, seq_len, word_dim]
            output = outputs.logits
            probs = output[:, 0]
            # output = output[:, :, :-1]  # truncate last token
            # candidate_id = labels[:, :, 1:]  # shift right
            # candidate_id = labels
            # cand_mask = candidate_id != model.module.pad_token_id
            # candidate_id = candidate_id.unsqueeze(-1)
            # candidate_id = labels
            cand_mask = labels != self.tokenizer.pad_token_id

            softmax_score, _output = torch.max(F.log_softmax(output, dim=-1), dim=-1)


            s = F.softmax(output, dim=-1)
            
            scores = []
            candidate_id = torch.where(labels != -100, labels, 3)
            for batch_size in range(s.shape[0]):
                seq_softmax = []
                for target_size in range(s.shape[1]):
                    target_index = candidate_id[batch_size, target_size]
                    seq_softmax.append(s[batch_size, target_size, target_index])
                scores.append(seq_softmax)
            scores = torch.tensor(scores)

            # print(softmax_score.shape, labels.shape)
            
            # scores = torch.gather(softmax_score, -1, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]

            cand_mask = cand_mask.float()
            # scores = torch.mul(scores, cand_mask).sum(-1) / ((cand_mask.sum(-1) + 0) ** 2.0) # [bz, cand_num]
            output = {'score': scores[:, 1:], "summary_score": scores[:, 0], "probs": probs}

            # # print(labels)
            # score_max_softmax, pred_softmax_label = torch.max(F.softmax(outputs.logits.view(-1, model.module.config.vocab_size), dim=1), axis=1)
            # # print(pred_softmax_label.shape)

            contrastive_loss = torch.log(RankingLoss(output['score'], output['summary_score']))

            loss += contrastive_loss


        return (loss, outputs) if return_outputs else loss