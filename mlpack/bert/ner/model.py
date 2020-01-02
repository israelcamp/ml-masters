from transformers.modeling_bert import BertForTokenClassification
from torchcrf import CRF
import torch


class BertForNERClassification(BertForTokenClassification):

    def __init__(self, config, weight_O=0.1, bias_O=None, pooler='last'):
        super().__init__(config)

        assert config.output_hidden_states
        del self.classifier

        num_labels = config.num_labels
        self.input_is_hidden_state = False

        self._build_pooler(pooler)
        self._build_classifier(config, pooler)
        if bias_O is not None:
            self.set_bias_tag_O(bias_O)

        assert isinstance(weight_O, float) and 0 < weight_O < 1
        weights = [weight_O] + [1.] * (num_labels - 1)
        weights = torch.tensor(weights)
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=weights)

    def _change_bert_grad(self, requires_grad):
        for name, params in self.model.named_parameters():
            if name.startswith('bert'):
                params.requires_grad = requires_grad

    @property
    def freeze_bert(self):
        self._change_bert_grad(False)

    @property
    def unfreeze_bert(self):
        self._change_bert_grad(True)

    def _build_pooler(self, pooler):
        assert pooler in ('last', 'sum', 'concat')
        if pooler == 'last':
            self.pooler = self.last_layer
        elif pooler == 'sum':
            self.pooler = self.sum_last_4_layers
        else:
            self.pooler = self.concat_last_4_layers

    def _build_classifier(self, config, pooler):
        """Build tag classifier."""
        if pooler in ('last', 'sum'):
            self.classifier = torch.nn.Linear(config.hidden_size,
                                              config.num_labels)
        else:
            assert pooler == 'concat'
            self.classifier = torch.nn.Linear(4 * config.hidden_size,
                                              config.num_labels)

    def set_bias_tag_O(self, bias_O):
        # Increase tag "O" bias to produce high probabilities early on and reduce
        # instability in early training
        if bias_O is not None:
            self.classifier.bias.data[0] = bias_O

    @staticmethod
    def sum_last_4_layers(sequence_outputs):
        """Sums the last 4 hidden representations of a sequence output of BERT.
        Args:
        -----
        sequence_output: Tuple of tensors of shape (batch, seq_length, hidden_size).
            For BERT base, the Tuple has length 13.

        Returns:
        --------
        summed_layers: Tensor of shape (batch, seq_length, hidden_size)
        """
        last_layers = sequence_outputs[-4:]
        return torch.stack(last_layers, dim=0).sum(dim=0)

    @staticmethod
    def last_layer(sequence_outputs):
        """Simply returns the last tensor of a list of tensors, indexing -1."""
        return sequence_outputs[-1]

    @staticmethod
    def concat_last_4_layers(sequence_outputs):
        last_layers = sequence_outputs[-4:]
        return torch.cat(last_layers, dim=-1)

    def hidden_state_as_input(self, use_hidden_state_as_input=False):
        self.input_is_hidden_state = use_hidden_state_as_input

    def bert_and_pool(self, input_ids, input_mask):
        if self.input_is_hidden_state:
            return input_ids
        else:
            _, _, hidden_states = self.bert(
                input_ids, attention_mask=input_mask)
            out = self.pooler(hidden_states)
            return out

    def predict_logits(self, input_ids, input_mask):
        out = self.bert_and_pool(input_ids, input_mask)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

    def active_logits_and_labels(self, logits, label_ids, label_mask=None):
        # take the active logits
        active_logits = logits.view(-1, self.num_labels)
        # take the active labels
        active_labels = label_ids.view(-1)
        if label_mask is not None:
            label_mask = label_mask.view(-1)
            active_logits = active_logits[label_mask == 1]
            active_labels = active_labels[label_mask == 1]
        return active_logits, active_labels

    def forward(self, input_ids, input_mask, label_ids=None, label_mask=None):
        logits = self.predict_logits(input_ids, input_mask)
        if label_ids is not None:

            active_logits, active_labels = self.active_logits_and_labels(
                logits, label_ids, label_mask)
            # calc the loss
            loss = self.loss_fct(active_logits, active_labels)

            return loss, active_logits, active_labels
        else:
            return logits


class BertForMaskedNERClassification(BertForNERClassification):

    def forward(self, input_ids, input_mask, label_ids=None, label_mask=None):
        logits = self.predict_logits(input_ids, input_mask)
        if label_ids is not None and label_mask is not None:
            # take the active logits
            label_mask = label_mask.view(-1)
            active_logits = logits.view(-1, self.num_labels)[label_mask == 1]

            # calc the loss
            loss = self.loss_fct(active_logits, label_ids)

            return loss, active_logits, label_ids
        return logits


class BertForSpanNERClassification(BertForNERClassification):

    def forward(self, input_ids, input_mask, label_ids=None, label_mask=None, max_context=None):
        out = self.bert_and_pool(input_ids, input_mask)
        hidden_size = out.shape[-1]
        if max_context is not None and label_ids is not None and label_mask is not None:
            _max_context = max_context.view(-1) == 1
            out = out.view(-1,
                           hidden_size)[_max_context].view(1, -1, hidden_size)
            out = self.dropout(out)
            out = self.classifier(out)
            # choose labels
            mask = label_mask.view(-1) == 1
            active_labels = label_ids.view(-1)[mask * _max_context]
            active_logits = out.view(-1,
                                     self.num_labels)[mask[_max_context]]
            # calc the loss
            loss = self.loss_fct(active_logits, active_labels)
            return loss, active_logits, active_labels
        return out


class BertCRF(BertForNERClassification):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        del self.loss_fct
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, input_mask, label_mask, label_ids=None):
        """Performs the forward pass of the network.

        If `labels` are not None, it will calculate and return the the loss,
        that is the negative log-likelihood of the batch.
        Otherwise, it will calculate the most probable sequence outputs using
        Viterbi decoding and return a list of sequences (List[List[int]]) of
        variable lengths."""
        outputs = {}

        logits = self.predict_logits(input_ids=input_ids,
                                     input_mask=input_mask)
        outputs['logits'] = logits

        # mask: mask padded sequence and also subtokens, because they must
        # not be used in CRF.
        mask = label_mask == 1
        batch_size = logits.shape[0]

        if label_ids is not None:
            # Negative of the log likelihood.
            # Loop through the batch here because of 2 reasons:
            # 1- the CRF package assumes the mask tensor cannot have interleaved
            # zeros and ones. In other words, the mask should start with True
            # values, transition to False at some moment and never transition
            # back to True. That can only happen for simple padded sequences.
            # 2- The first column of mask tensor should be all True, and we
            # cannot guarantee that because we have to mask all non-first
            # subtokens of the WordPiece tokenization.
            loss = 0
            for seq_logits, seq_labels, seq_mask in zip(logits, label_ids, mask):

                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                seq_labels = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(seq_logits, seq_labels,
                                 reduction='token_mean')

            loss /= batch_size
            active_logits, active_labels = self.active_logits_and_labels(
                logits, label_ids, mask)
            outputs['loss'] = loss
            outputs['active_logits'] = active_logits
            outputs['active_labels'] = active_labels

        else:
            # Same reasons for iterating
            output_tags = []
            for seq_logits, seq_mask in zip(logits, mask):
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                tags = self.crf.decode(seq_logits)
                # Unpack "batch" results
                output_tags.append(tags[0])

            outputs['y_pred'] = output_tags

        return outputs
