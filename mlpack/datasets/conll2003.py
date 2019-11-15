import os
import torch
from torch.utils.data import Dataset


class CoNLL2003Dataset(Dataset):

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        return torch.tensor(feat.input_ids), torch.tensor(feat.input_mask), \
            torch.tensor(feat.label_id) - 1, torch.tensor(feat.label_mask)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, masked_word=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.masked_word = masked_word


def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """
            here "X" used to represent "##eer","##soo" and so on!
            "[PAD]" for padding
            :return:
        """
        return ["[PAD]", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]", "X"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, sep_tag='X'):
    """Loads a data file into a list of `InputBatch`s."""

    assert sep_tag in ('X', 'same')

    def put_sep_tag(tag):
        return (sep_tag, 0) if sep_tag == 'X' else (tag if tag == 'O' else 'I-'+tag.split('-')[-1], 1)

    label_map = {label: i for i, label in enumerate(label_list, 0)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    # TODO: Fix here, in case the sep_tag is same then the label_mask should be 1
                    # could also give this as argument for the function
                    tag, mask = put_sep_tag(label_1)
                    labels.append(tag)
                    valid.append(0)
                    label_mask.append(mask)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        # adding the cls
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 0)  # ignoring [CLS] on loss
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        # adding the sep
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(0)  # ignoring [SEP] on loss
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features


def convert_example_to_masked_feature(textlist, label, label_map, max_seq_length, tokenizer, masked_index):

    tokens = []
    label_mask = []
    for i, word in enumerate(textlist):
        if i == masked_index:  # mask the word
            token = ['[MASK]']
            masked_word = word
            label_mask.append(1)
        else:
            token = tokenizer.tokenize(word)
            label_mask += len(token) * [0]
        tokens.extend(token)

    if len(tokens) >= max_seq_length - 1:  # cutting the sequence
        tokens = tokens[0:(max_seq_length - 2)]
        label_mask = label_mask[0:(max_seq_length - 2)]

    if all([l == 0 for l in label_mask]):
        return None
    # adding the [CLS]
    tokens.insert(0, '[CLS]')
    label_mask.insert(0, 0)

    # adding the [SEP]
    tokens.append('[SEP]')
    label_mask.append(0)

    # starting input_mask
    input_mask = len(tokens) * [1]

    # adding the [PAD] tokens
    missing = max_seq_length - len(tokens)
    if missing > 0:
        tokens += missing * ['[PAD]']
        label_mask += missing * [0]
        input_mask += missing * [0]

    assert all([len(v) == max_seq_length for v in [
               tokens, input_mask, label_mask]])

    # doing input_ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=None,
                         label_id=label_map[label],
                         valid_ids=None,
                         label_mask=label_mask,
                         masked_word=masked_word
                         )


def convert_examples_to_features_masked(examples, label_list, max_seq_length, tokenizer):

    label_map = {label: i for i, label in enumerate(label_list, 0)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        for j in range(len(textlist)):
            label = example.label[j]  # take the label of the word
            feat = convert_example_to_masked_feature(
                textlist, label, label_map, max_seq_length, tokenizer, j)
            if feat is not None:
                features.append(feat)
    return features


def get_conll2003(dir_path):
    processor = NerProcessor()
    train_examples = processor.get_train_examples(dir_path)
    valid_examples = processor.get_dev_examples(dir_path)
    test_examples = processor.get_test_examples(dir_path)
    return {
        'train': train_examples,
        'valid': valid_examples,
        'test': test_examples,
    }, processor.get_labels()


def get_conll2003_features(examples_dict, labels, max_seq_length, tokenizer, sep_tag='X'):
    return {
        k: convert_examples_to_features(
            examples, labels, max_seq_length, tokenizer, sep_tag) for k, examples in examples_dict.items()
    }
