import torch

from mlpack.bert.features import convert_examplewithtokens_to_spanfeatures
from mlpack.bert.examples import text_to_example_with_tokens


class Entity:
    def __init__(self, label, text, start, end):
        self.label, self.text, self.start, self.end = label, text, start, end

    @property
    def span(self):
        return self.start, self.end

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()


class BertNERHandler:

    def __init__(self, model, tokenizer, labels):
        self.model, self.tokenizer = model, tokenizer
        self.labels = labels
        self.all_labels = ['[PAD]'] + self.labels + \
            ['[CLS]', '[SEP]', 'X', '[UNK]']

        self.model.to(self.device)

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def ent_names(self):
        return list(set([
            l.split('-')[-1] for l in self.labels if l not in ('O')
        ]))

    def _prepare_tensors(self, feature):
        device = self.device
        input_ids, input_mask, label_mask = feature.input_ids, feature.input_mask, feature.label_mask
        input_ids = torch.tensor(input_ids).to(device).view(1, -1)
        input_mask = torch.tensor(input_mask).to(device).view(1, -1)
        label_mask = torch.tensor(label_mask).to(device).view(1, -1)

        return input_ids, input_mask, label_mask

    def convert_output_to_lists(self, out, input_ids, label_mask):
        data = []
        # for ex, in_ids, lm in zip(out, input_ids, label_mask):
        preds = out[0].argmax(1).to('cpu').numpy().tolist()
        in_ids = input_ids[0].to('cpu').numpy().tolist()
        sep_index = in_ids.index(103)  # [SEP] id
        token_ids = in_ids[1:sep_index]
        labels = [
            self.labels[y] for y in preds[1:sep_index]
        ]
        mask = label_mask[0].to('cpu').numpy().tolist()[1:sep_index]
        # data.append(
        return (token_ids, labels, mask)
        # )
        # return data[0]

    def find_ient(self, token_ids, labels, mask, ent_name, i):
        name = []
        if labels[i] == 'I-' + ent_name:
            try:
                j = i + mask[i + 1:].index(1) + 1
            except ValueError:
                j = len(mask)
                name += token_ids[i:j]
                return name, j
            name += token_ids[i:j]
            n, j = self.find_ient(token_ids, labels, mask, ent_name, j)
            name += n
        else:
            j = i
        return name, j

    def get_ents(self, token_ids, labels, mask, ent_name):
        ents = []
        for i, (t, l, m) in enumerate(zip(token_ids, labels, mask)):
            if m == 0:
                continue
            if l == 'B-' + ent_name:
                try:
                    j = i + mask[i + 1:].index(1) + 1
                except ValueError:
                    j = len(mask)
                    name = token_ids[i:j]
                else:
                    name = []
                    j = i + mask[i + 1:].index(1) + 1
                    name += token_ids[i:j]
                    n, j = self.find_ient(token_ids,
                                          labels, mask, ent_name, j)
                    name += n
                finally:
                    entity = Entity(
                        ent_name, self.tokenizer.decode(name), i + 1, j + 1)  # +1 to account for removing [CLS]
                    ents.append(entity)
        return ents

    def get_all_ents(self, token_ids, labels, mask):
        ents = {}
        for ent_name in self.ent_names:
            ents[ent_name] = self.get_ents(token_ids, labels, mask, ent_name)
        return ents

    def __call__(self, feature):
        input_ids, input_mask, label_mask = self._prepare_tensors(feature)
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids, input_mask)
        token_ids, labels, mask = self.convert_output_to_lists(
            out, input_ids, label_mask)
        ents = self.get_all_ents(token_ids, labels, mask)
        return ents


class BertNERSpanHandler(BertNERHandler):

    @staticmethod
    def remove_non_max_context_entities(entities, feature):
        ents = {}
        for ent_name, ent_values in entities.items():
            keep_ents = [
                # and feature.token_is_max_context[e.end - 1]
                e for e in ent_values if feature.token_is_max_context[e.start]
            ]
            ents[ent_name] = keep_ents
        return ents

    @staticmethod
    def add_start_and_end_to_ent_on_org_text(entities, feature, word_to_char_offset):
        for ent_name, ents in entities.items():
            for e in ents:
                start_idx = feature.token_to_orig_map[e.start]
                end_idx = feature.token_to_orig_map[e.end - 1]

                start_char = word_to_char_offset[start_idx][0]
                end_char = word_to_char_offset[end_idx][-1]

                e.start_char = start_char
                e.end_char = end_char
        return entities

    @staticmethod
    def merge_dicts(dict_one, dict_two):
        for key, values in dict_one.items():
            if key in dict_two:
                dict_two[key] += values
            else:
                dict_two[key] = values
        return dict_two

    @staticmethod
    def create_all_token_to_orig_map(features):
        i = 1
        all_token_to_orig_map = {}
        for feature in features:
            token_is_max_context = feature.token_is_max_context
            for idx, is_max in token_is_max_context.items():
                if is_max:
                    all_token_to_orig_map[i] = feature.token_to_orig_map[idx]
                    i += 1
        return all_token_to_orig_map

    def create_feature_from_text(self, text):
        example = text_to_example_with_tokens(text)
        features, word_to_char_offset = convert_examplewithtokens_to_spanfeatures(
            example, self.all_labels, 512, self.tokenizer, 256, False)
        return features, word_to_char_offset

    def from_feature(self, feature):
        input_ids, input_mask, label_mask = self._prepare_tensors(feature)
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids, input_mask)
        token_ids, labels, mask = self.convert_output_to_lists(
            out, input_ids, label_mask)
        ents = {}
        for ent_name in self.ent_names:
            ents[ent_name] = self.get_ents(token_ids, labels, mask, ent_name)
        return ents

    def __call__(self, text, remove_non_max_context=True):
        features, word_to_char_offset = self.create_feature_from_text(text)
        entities = [
            BertNERHandler.__call__(self, feature) for feature in features
        ]
        final_ents = {}
        for ents, feature in zip(entities, features):
            if remove_non_max_context:
                ents = BertNERSpanHandler.remove_non_max_context_entities(
                    ents, feature)
            ents = BertNERSpanHandler.add_start_and_end_to_ent_on_org_text(
                ents, feature, word_to_char_offset)
            # final_ents.append(ents)
            final_ents = BertNERSpanHandler.merge_dicts(ents, final_ents)
        return final_ents
