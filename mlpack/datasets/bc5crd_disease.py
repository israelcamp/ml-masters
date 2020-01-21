import os

from mlpack.datasets.conll2003 import DataProcessor, InputExample, InputFeatures, CoNLL2003Dataset as BC5CDRDataset
from mlpack.datasets.conll2003 import convert_example_to_masked_feature, convert_examples_to_features
from mlpack.datasets.conll2003 import convert_examples_to_features_masked


class NerProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "devel.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """
            here "X" used to represent "##eer","##soo" and so on!
            "[PAD]" for padding
            :return:
        """
        return ["[PAD]", "O", "B-DIS", "I-DIS", "[CLS]", "[SEP]", "X"]

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

    def _read_tsv(self, filename):
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
            splits = line.split('\t')
            sentence.append(splits[0])
            label.append(splits[-1]+'-DIS')

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
        return data


def get_bc5cdr_disease(dir_path):
    processor = NerProcessor()
    train_examples = processor.get_train_examples(dir_path)
    valid_examples = processor.get_dev_examples(dir_path)
    test_examples = processor.get_test_examples(dir_path)
    return {
        'train': train_examples,
        'valid': valid_examples,
        'test': test_examples,
    }, processor.get_labels()


def get_bc5cdr_disease_features(examples_dict, labels, max_seq_length, tokenizer, sep_tag='X'):
    return {
        k: convert_examples_to_features(
            examples, labels, max_seq_length, tokenizer, sep_tag) for k, examples in examples_dict.items()
    }
