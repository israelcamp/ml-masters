import os
import collections

from tqdm import tqdm


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, valid_ids=None, label_mask=None, masked_word=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.masked_word = masked_word

    def to_dict(self):
        return {
            'input_ids': self.input_ids,
            'input_mask': self.input_mask,
            'segment_ids': self.segment_ids,
            'label_ids': self.label_ids,
            'valid_ids': self.valid_ids,
            'label_mask': self.label_mask,
            'masked_word': self.masked_word
        }


class InputSpanFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 label_ids,
                 label_mask):
        self.unique_id = unique_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_ids = label_ids
        self.label_mask = label_mask

    def to_dict(self):
        return {
            'unique_id': self.unique_id,
            'doc_span_index': self.doc_span_index,
            'tokens': self.tokens,
            'token_to_orig_map': self.token_to_orig_map,
            'token_is_max_context': self.token_is_max_context,
            'input_ids': self.input_ids,
            'input_mask': self.input_mask,
            'label_ids': self.label_ids,
            'label_mask': self.label_mask
        }


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, sep_tag='X'):
    """Loads a data file into a list of `InputBatch`s."""

    assert sep_tag in ('X', 'same')

    def put_sep_tag(tag):
        return (sep_tag, 0) if sep_tag == 'X' else (tag if tag == 'O' else 'I-' + tag.split('-')[-1], 1)

    label_map = {label: i for i, label in enumerate(label_list, 0)}

    features = []
    for (ex_index, example) in tqdm(enumerate(examples), total=len(examples)):
        textlist = example.text.split(' ')
        labellist = example.labels
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
                          label_ids=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_examplewithtokens_to_spanfeatures(example,
                                              label_list, max_seq_length, tokenizer,
                                              doc_stride, training=False):
    """Loads a data file into a list of `InputBatch`s."""
    '''I still need to fix the example'''

    features = []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    all_doc_labels = []
    all_doc_labels_mask = []
    doc_text, char_to_word_offset = example.tokens, example.char_to_word_offset
    word_to_char_offset = {
        v: [i for i, q in enumerate(char_to_word_offset) if q == v] for v in list(set(char_to_word_offset))
    }

    doc_labels = example.labels if training else ['[UNK]'] * len(doc_text)
    assert len(doc_labels) == len(doc_text)

    for (i, token) in enumerate(doc_text):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token.text)
        token_label = label_list.index(doc_labels[i])
        for j, sub_token in enumerate(sub_tokens):
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
            if j == 0:
                all_doc_labels.append(token_label)
                all_doc_labels_mask.append(1)
            else:
                all_doc_labels.append(label_list.index('X'))
                all_doc_labels_mask.append(0)

    # The -2 accounts for [CLS] and [SEP]
    max_tokens_for_doc = max_seq_length - 2

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    unique_count = 0
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        label_ids = []
        label_mask = []
        token_to_orig_map = {}
        token_is_max_context = {}
        # insert [CLS]
        tokens.append("[CLS]")
        label_ids.append(label_list.index("[CLS]"))
        label_mask.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(
                tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            label_ids.append(all_doc_labels[split_token_index])
            label_mask.append(all_doc_labels_mask[split_token_index])

        tokens.append("[SEP]")
        label_ids.append(label_list.index("[SEP]"))
        label_mask.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(label_list.index("[PAD]"))
            label_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(label_mask) == max_seq_length

        feature = InputSpanFeatures(
            unique_id=f'{example.id}_{unique_count}',
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            label_ids=label_ids,
            label_mask=label_mask,
        )

        unique_count += 1

        features.append(feature)

    return features, word_to_char_offset


def convert_example_to_spanfeatures(example, label_list, max_seq_length, tokenizer,
                                    doc_stride, training=False):
    """Loads a data file into a list of `InputBatch`s."""
    '''I still need to fix the example'''

    features = []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    all_doc_labels = []
    all_doc_labels_mask = []
    doc_text = example.text.split(' ')
    doc_labels = example.labels if training else ['[UNK]'] * len(doc_text)
    assert len(doc_labels) == len(doc_text)

    for (i, token) in enumerate(doc_text):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        token_label = label_list.index(doc_labels[i])
        for j, sub_token in enumerate(sub_tokens):
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
            if j == 0:
                all_doc_labels.append(token_label)
                all_doc_labels_mask.append(1)
            else:
                all_doc_labels.append(label_list.index('X'))
                all_doc_labels_mask.append(0)

    # The -2 accounts for [CLS] and [SEP]
    max_tokens_for_doc = max_seq_length - 2

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    unique_count = 0
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        label_ids = []
        label_mask = []
        token_to_orig_map = {}
        token_is_max_context = {}
        # insert [CLS]
        tokens.append("[CLS]")
        label_ids.append(label_list.index("[CLS]"))
        label_mask.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(
                tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            label_ids.append(all_doc_labels[split_token_index])
            label_mask.append(all_doc_labels_mask[split_token_index])

        tokens.append("[SEP]")
        label_ids.append(label_list.index("[SEP]"))
        label_mask.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(label_list.index("[PAD]"))
            label_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(label_mask) == max_seq_length

        feature = InputSpanFeatures(
            unique_id=f'{example.id}_{unique_count}',
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            label_ids=label_ids,
            label_mask=label_mask,
        )

        unique_count += 1

        features.append(feature)

    return features
