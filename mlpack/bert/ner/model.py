from transformers.modeling_bert import BertForTokenClassification
import torch


class BertForNERClassification(BertForTokenClassification):

    def __init__(self, config, weight_O=0.1, bias_O=None, pooler='last'):
        super().__init__(config)

        assert config.output_hidden_states
        del self.classifier

        num_labels = config.num_labels

        self._build_pooler(pooler)
        self._build_classifier(config, pooler)
        if bias_O is not None:
            self.set_bias_tag_O(bias_O)

        assert isinstance(weight_O, float) and 0 < weight_O < 1
        weights = [weight_O] + [1.] * (num_labels - 1)
        weights = torch.tensor(weights)
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=weights)

    def _build_pooler(self, pooler):
        if pooler == 'last':
            self.pooler = self.last_layer
        else:
            assert pooler == 'sum'
            self.pooler = self.sum_last_4_layers

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

    def forward(self, input_ids, input_mask, label_ids, label_mask):

        _, _, hidden_states = self.bert(input_ids, attention_mask=input_mask)

        out = self.pooler(hidden_states)

        out = self.dropout(out)
        out = self.classifier(out)
        # take the active logits
        label_mask = label_mask.view(-1)
        active_logits = out.view(-1, self.num_labels)[label_mask == 1]

        # take the active labels
        # remove one because of the [PAD] being the 0
        active_labels = label_ids.view(-1)[label_mask == 1] - 1

        # calc the loss
        loss = self.loss_fct(active_logits, active_labels)

        return loss, active_logits, active_labels


class BertForMaskedNERClassification(BertForNERClassification):

    def forward(self, input_ids, input_mask, label_ids, label_mask):

        _, _, hidden_states = self.bert(input_ids, attention_mask=input_mask)

        out = self.pooler(hidden_states)

        out = self.dropout(out)
        out = self.classifier(out)
        # take the active logits
        label_mask = label_mask.view(-1)
        active_logits = out.view(-1, self.num_labels)[label_mask == 1]

        # calc the loss
        loss = self.loss_fct(active_logits, label_ids)

        return loss, active_logits, label_ids
