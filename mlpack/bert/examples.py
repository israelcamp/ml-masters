import re
import os
from typing import List

from tokenization import TokenizerWithAlignment


class InputExample:
    """A single training/test example for named entity recognition."""

    def __init__(self, id, text, labels):
        """Constructs a InputExample.

        Args:
            id: Unique id for the example.
            text(string): The untokenized text of the first sequence.
            label(list(string)): The label for every word in text
        """
        self.id = id
        self.text = text
        self.labels = labels


class InputExampleWithTokens:
    def __init__(self, id, tokens, labels, char_to_word_offset, orig_text=None, sep=''):
        self.id, self.tokens, self.labels, self.sep = id, tokens, labels, sep
        self.char_to_word_offset = char_to_word_offset
        self.orig_text = orig_text


def prepare_text(text: str) -> str:
    t = text.strip().replace('\n', ' ').replace(
        ':', ' : ').replace('. ', ' . ').replace(', ', ' , ').replace(
        '(', ' ( ').replace(')', ' ) ').replace('[', '[ ').replace(']', ' ] ')
    t = t.replace('-', ' - ')
    return re.sub(' {2,}', ' ', t)


def text_to_example(text: str) -> InputExample:
    example = InputExample(
        text=text,
        labels=['[UNK]'] * len(text.split(' ')),
        id=None
    )
    return example


def text_to_example_with_tokens(text: str) -> InputExampleWithTokens:
    tokenizer = TokenizerWithAlignment()
    tokens, char_to_word = tokenizer(text)
    example = InputExampleWithTokens(
        orig_text=text,
        tokens=tokens,
        labels=['[UNK]'] * len(tokens),
        char_to_word_offset=char_to_word,
        id=None
    )
    return example


def texts_to_examples(texts: List[str], do_prepare_text=False) -> List[InputExample]:
    return [
        text_to_example(prepare_text(t)) if do_prepare_text else text_to_example(t) for t in texts
    ]


def read_file(filepath):
    with open(filepath) as f:
        return f.read()
