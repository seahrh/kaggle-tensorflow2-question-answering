# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT-joint baseline for NQ v1.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import json
import os
import random
import re

import enum
import bert_tokenization as tokenization
import numpy as np
import tensorflow as tf

from nq_flags import DEFAULT_FLAGS as FLAGS

TextSpan = collections.namedtuple("TextSpan", "token_positions text")


class AnswerType(enum.IntEnum):
    """Type of NQ answer."""
    UNKNOWN = 0
    YES = 1
    NO = 2
    SHORT = 3
    LONG = 4


class Answer(collections.namedtuple("Answer", ["type", "text", "offset"])):
    """Answer record.

    An Answer contains the type of the answer and possibly the text (for
    long) as well as the offset (for extractive).
    """

    def __new__(cls, type_, text=None, offset=None):
        return super(Answer, cls).__new__(cls, type_, text, offset)


class NqExample(object):
    """A single training/test example."""

    def __init__(self,
                             example_id,
                             qas_id,
                             questions,
                             doc_tokens,
                             doc_tokens_map=None,
                             answer=None,
                             start_position=None,
                             end_position=None):
        self.example_id = example_id
        self.qas_id = qas_id
        self.questions = questions
        self.doc_tokens = doc_tokens
        self.doc_tokens_map = doc_tokens_map
        self.answer = answer
        self.start_position = start_position
        self.end_position = end_position


def has_long_answer(a):
    return (a["long_answer"]["start_token"] >= 0 and
                    a["long_answer"]["end_token"] >= 0)


def should_skip_context(e, idx):
    if (
        FLAGS.skip_nested_contexts and
            not e["long_answer_candidates"][idx]["top_level"]):
        return True
    elif not get_candidate_text(e, idx).text.strip():
        # Skip empty contexts.
        return True
    else:
        return False


def get_first_annotation(e):
    """Returns the first short or long answer in the example.

    Args:
        e: (dict) annotated example.

    Returns:
        annotation: (dict) selected annotation
        annotated_idx: (int) index of the first annotated candidate.
        annotated_sa: (tuple) char offset of the start and end token
                of the short answer. The end token is exclusive.
    """

    # On Kaggle, the test dataset has no 'annotations' attribute.
    if "annotations" not in e:
        return None, -1, (-1, -1)

    positive_annotations = sorted(
            [a for a in e["annotations"] if has_long_answer(a)],
            key=lambda a: a["long_answer"]["candidate_index"])

    for a in positive_annotations:
        if a["short_answers"]:
            idx = a["long_answer"]["candidate_index"]
            start_token = a["short_answers"][0]["start_token"]
            end_token = a["short_answers"][-1]["end_token"]
            return a, idx, (token_to_char_offset(e, idx, start_token),
                                            token_to_char_offset(e, idx, end_token) - 1)

    for a in positive_annotations:
        idx = a["long_answer"]["candidate_index"]
        return a, idx, (-1, -1)

    return None, -1, (-1, -1)


def get_text_span(example, span):
    """Returns the text in the example's document in the given token span."""
    token_positions = []
    tokens = []
    for i in range(span["start_token"], span["end_token"]):
        t = example["document_tokens"][i]
        if not t["html_token"]:
            token_positions.append(i)
            token = t["token"].replace(" ", "")
            tokens.append(token)
    return TextSpan(token_positions, " ".join(tokens))


def token_to_char_offset(e, candidate_idx, token_idx):
    """Converts a token index to the char offset within the candidate."""
    c = e["long_answer_candidates"][candidate_idx]
    char_offset = 0
    for i in range(c["start_token"], token_idx):
        t = e["document_tokens"][i]
        if not t["html_token"]:
            token = t["token"].replace(" ", "")
            char_offset += len(token) + 1
    return char_offset


def get_candidate_type(e, idx):
    """Returns the candidate's type: Table, Paragraph, List or Other."""
    c = e["long_answer_candidates"][idx]
    first_token = e["document_tokens"][c["start_token"]]["token"]
    if first_token == "<Table>":
        return "Table"
    elif first_token == "<P>":
        return "Paragraph"
    elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
        return "List"
    elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
        return "Other"
    else:
        tf.logging.warning("Unknoww candidate type found: %s", first_token)
        return "Other"


def add_candidate_types_and_positions(e):
    """Adds type and position info to each candidate in the document."""
    counts = collections.defaultdict(int)
    for idx, c in candidates_iter(e):
        context_type = get_candidate_type(e, idx)
        if counts[context_type] < FLAGS.max_position:
            counts[context_type] += 1
        c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])


def get_candidate_type_and_position(e, idx):
    """Returns type and position info for the candidate at the given index."""
    if idx == -1:
        return "[NoLongAnswer]"
    else:
        return e["long_answer_candidates"][idx]["type_and_position"]


def get_candidate_text(e, idx):
    """Returns a text representation of the candidate at the given index."""
    # No candidate at this index.
    if idx < 0 or idx >= len(e["long_answer_candidates"]):
        return TextSpan([], "")

    # This returns an actual candidate.
    return get_text_span(e, e["long_answer_candidates"][idx])


def candidates_iter(e):
    """Yield's the candidates that should not be skipped in an example."""
    for idx, c in enumerate(e["long_answer_candidates"]):
        if should_skip_context(e, idx):
            continue
        yield idx, c

        
def make_nq_answer(contexts, answer):
    """Makes an Answer object following NQ conventions.

    Args:
        contexts: string containing the context
        answer: dictionary with `span_start` and `input_text` fields

    Returns:
        an Answer object. If the Answer type is YES or NO or LONG, the text
        of the answer is the long answer. If the answer type is UNKNOWN, the text of
        the answer is empty.
    """
    start = answer["span_start"]
    end = answer["span_end"]
    input_text = answer["input_text"]

    if (answer["candidate_id"] == -1 or start >= len(contexts) or
            end > len(contexts)):
        answer_type = AnswerType.UNKNOWN
        start = 0
        end = 1
    elif input_text.lower() == "yes":
        answer_type = AnswerType.YES
    elif input_text.lower() == "no":
        answer_type = AnswerType.NO
    elif input_text.lower() == "long":
        answer_type = AnswerType.LONG
    else:
        answer_type = AnswerType.SHORT

    return Answer(answer_type, text=contexts[start:end], offset=start)


def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #    Doc: the man went to the store and bought a gallon of milk
    #    Span A: the man went to the
    #    Span B: to the store and bought
    #    Span C: and bought a gallon of
    #    ...
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
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


# A special token in NQ is made of non-space chars enclosed in square brackets.
_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)


def tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.

    Args:
        tokenizer: a tokenizer from bert.tokenization.FullTokenizer
        text: text to tokenize
        apply_basic_tokenization: If True, apply the basic tokenization. If False,
            apply the full tokenization (basic + wordpiece).

    Returns:
        tokenized text.

    A special token is any text with no spaces enclosed in square brackets with no
    space, so we separate those out and look them up in the dictionary before
    doing actual tokenization.
    """
    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize
    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.vocab:
                tokens.append(token)
            else:
                tokens.append(tokenizer.wordpiece_tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))
    return tokens


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                             unique_id,
                             example_index,
                             doc_span_index,
                             tokens,
                             token_to_orig_map,
                             token_is_max_context,
                             input_ids,
                             input_mask,
                             segment_ids,
                             start_position=None,
                             end_position=None,
                             answer_text="",
                             answer_type=AnswerType.SHORT):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.answer_text = answer_text
        self.answer_type = answer_type


def nq_line_to_nq_entry(line):
    """Creates an NQ example (actually, NQ entry) from a given line of JSON."""
    e = json.loads(line, object_pairs_hook=collections.OrderedDict)
    e["document_tokens"] = []
    for token in e["document_text"].split(" "):
        doc_token = {"token": token, "html_token": token.startswith("<") and token.endswith(">")}
        e["document_tokens"].append(doc_token)
    
    add_candidate_types_and_positions(e)
    annotation, annotated_idx, annotated_sa = get_first_annotation(e)

    # annotated_idx: index of the first annotated context, -1 if null.
    # annotated_sa: short answer start and end char offsets, (-1, -1) if null.
    question = {"input_text": e["question_text"]}
    answer = {
            "candidate_id": annotated_idx,
            "span_text": "",
            "span_start": -1,
            "span_end": -1,
            "input_text": "long",
    }

    # Yes/no answers are added in the input text.
    if annotation is not None:
        assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
        if annotation["yes_no_answer"] in ("YES", "NO"):
            answer["input_text"] = annotation["yes_no_answer"].lower()

    # Add a short answer if one was found.
    if annotated_sa != (-1, -1):
        answer["input_text"] = "short"
        span_text = get_candidate_text(e, annotated_idx).text
        answer["span_text"] = span_text[annotated_sa[0]:annotated_sa[1]]
        answer["span_start"] = annotated_sa[0]
        answer["span_end"] = annotated_sa[1]
        expected_answer_text = get_text_span(
                e, {
                        "start_token": annotation["short_answers"][0]["start_token"],
                        "end_token": annotation["short_answers"][-1]["end_token"],
                }).text
        assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                                                                                 answer["span_text"])

    # Add a long answer if one was found.
    elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
        answer["span_text"] = get_candidate_text(e, annotated_idx).text
        answer["span_start"] = 0
        answer["span_end"] = len(answer["span_text"])

    context_idxs = [-1]
    context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
    context_list[-1]["text_map"], context_list[-1]["text"] = (
            get_candidate_text(e, -1))
    for idx, _ in candidates_iter(e):
        context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
        context["text_map"], context["text"] = get_candidate_text(e, idx)
        context_idxs.append(idx)
        context_list.append(context)
        if len(context_list) >= FLAGS.max_contexts:
            break

    # Assemble example.
    example = {
            "id": str(e["example_id"]),
            "questions": [question],
            "answers": [answer],
            "has_correct_context": annotated_idx in context_idxs
    }

    single_map = []
    single_context = []
    offset = 0
    for context in context_list:
        single_map.extend([-1, -1])
        single_context.append("[ContextId=%d] %s" %
                                                    (context["id"], context["type"]))
        offset += len(single_context[-1]) + 1
        if context["id"] == annotated_idx:
            answer["span_start"] += offset
            answer["span_end"] += offset

        # Many contexts are empty once the HTML tags have been stripped, so we
        # want to skip those.
        if context["text"]:
            single_map.extend(context["text_map"])
            single_context.append(context["text"])
            offset += len(single_context[-1]) + 1

    example["contexts"] = " ".join(single_context)
    example["contexts_map"] = single_map
    if annotated_idx in context_idxs:
        expected = example["contexts"][answer["span_start"]:answer["span_end"]]

        # This is a sanity check to ensure that the calculated start and end
        # indices match the reported span text. If this assert fails, it is likely
        # a bug in the data preparation code above.
        assert expected == answer["span_text"], (expected, answer["span_text"])

    return example


def nq_entry_to_nq_examples(entry, is_training):
    """Converts a NQ entry into a list of NqExamples."""

    def is_whitespace(c):
        return c in " \t\r\n" or ord(c) == 0x202F

    examples = []
    contexts_id = entry["id"]
    contexts = entry["contexts"]
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in contexts:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    questions = []
    for i, question in enumerate(entry["questions"]):
        qas_id = "{}".format(contexts_id)
        question_text = question["input_text"]
        start_position = None
        end_position = None
        answer = None
        if is_training:
            answer_dict = entry["answers"][i]
            answer = make_nq_answer(contexts, answer_dict)

            # For now, only handle extractive, yes, and no.
            if answer is None or answer.offset is None:
                continue
            start_position = char_to_word_offset[answer.offset]
            end_position = char_to_word_offset[answer.offset + len(answer.text) - 1]

            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                    tokenization.whitespace_tokenize(answer.text))
            if actual_text.find(cleaned_answer_text) == -1:
                tf.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text,
                                                     cleaned_answer_text)
                continue

        questions.append(question_text)
        example = NqExample(
                example_id=int(contexts_id),
                qas_id=qas_id,
                questions=questions[:],
                doc_tokens=doc_tokens,
                doc_tokens_map=entry.get("contexts_map", None),
                answer=answer,
                start_position=start_position,
                end_position=end_position)
        examples.append(example)
    return examples


def single_nq_example_to_nq_features(example, tokenizer, is_training):
    """Converts a single NqExample into a list of InputFeatures."""
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    features = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenize(tokenizer, token)
        tok_to_orig_index.extend([i] * len(sub_tokens))
        all_doc_tokens.extend(sub_tokens)

    # `tok_to_orig_index` maps wordpiece indices to indices of whitespace
    # tokenized word tokens in the contexts. The word tokens might themselves
    # correspond to word tokens in a larger document, with the mapping given
    # by `doc_tokens_map`.
    if example.doc_tokens_map:
        tok_to_orig_index = [
                example.doc_tokens_map[index] for index in tok_to_orig_index
        ]

    # QUERY
    query_tokens = []
    query_tokens.append("[Q]")
    query_tokens.extend(tokenize(tokenizer, example.questions[-1]))
    if len(query_tokens) > FLAGS.max_query_length:
        query_tokens = query_tokens[-FLAGS.max_query_length:]

    # ANSWER
    tok_start_position = 0
    tok_end_position = 0
    if is_training:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = FLAGS.max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(    # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        length = min(length, max_tokens_for_doc)
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, FLAGS.doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        tokens.extend(query_tokens)
        segment_ids.extend([0] * len(query_tokens))
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = check_is_max_context(doc_spans, doc_span_index,
                                                                                        split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        assert len(tokens) == len(segment_ids)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (FLAGS.max_seq_length - len(input_ids))
        input_ids.extend(padding)
        input_mask.extend(padding)
        segment_ids.extend(padding)

        assert len(input_ids) == FLAGS.max_seq_length
        assert len(input_mask) == FLAGS.max_seq_length
        assert len(segment_ids) == FLAGS.max_seq_length

        start_position = None
        end_position = None
        answer_type = None
        answer_text = ""
        if is_training:
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            contains_an_annotation = (
                    tok_start_position >= doc_start and tok_end_position <= doc_end)
            if ((not contains_an_annotation) or
                    example.answer.type == AnswerType.UNKNOWN):
                # If an example has unknown answer type or does not contain the answer
                # span, then we only include it with probability --include_unknowns.
                # When we include an example with unknown answer type, we set the first
                # token of the passage to be the annotated short span.
                if (FLAGS.include_unknowns < 0 or
                        random.random() > FLAGS.include_unknowns):
                    continue
                start_position = 0
                end_position = 0
                answer_type = AnswerType.UNKNOWN
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
                answer_type = example.answer.type

            answer_text = " ".join(tokens[start_position:(end_position + 1)])

        feature = InputFeatures(
                unique_id=-1,
                example_index=-1,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                answer_text=answer_text,
                answer_type=answer_type)

        features.append(feature)

    return features


def nq_examples_to_nq_features(examples, tokenizer, is_training):
    """Converts a list of NqExamples into InputFeatures."""

    all_features = []

    for example in examples:
        example_index = example.example_id
        features = single_nq_example_to_nq_features(example, tokenizer, is_training)

        for feature in features:
            feature.example_index = example_index
            feature.unique_id = feature.example_index + feature.doc_span_index

            all_features.append(feature)

    return all_features


def nq_line_to_nq_examples(nq_line, is_training):

    nq_entry = nq_line_to_nq_entry(nq_line)
    nq_examples = nq_entry_to_nq_examples(nq_entry, is_training)

    return nq_examples


def nq_line_to_nq_features(nq_line, tokenizer, is_training):

    nq_examples = nq_line_to_nq_examples(nq_line, is_training)
    nq_features = nq_examples_to_nq_features(nq_examples, tokenizer, is_training)

    return nq_features


class TFExampleCreator(object):
    """Functor for creating NQ tf.Examples."""

    def __init__(self, is_training):
        
        self.is_training = is_training
        self.tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        print(FLAGS.include_unknowns)
        print(FLAGS.max_seq_length)

    def process_nq_entry(self, nq_entry, collected_nq_features):
        """Coverts an NQ entry into a list of serialized tf examples."""
        
        nq_examples = nq_entry_to_nq_examples(nq_entry, self.is_training)
        nq_features = nq_examples_to_nq_features(nq_examples, self.tokenizer, self.is_training)

        for nq_feature in nq_features:

            if collected_nq_features is not None:
                collected_nq_features.append(nq_feature)

            nq_feature.example_index = int(nq_entry["id"])
            nq_feature.unique_id = (
                    nq_feature.example_index + nq_feature.doc_span_index)

            def create_int_feature(values):
                return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

            features = collections.OrderedDict()
            features["unique_ids"] = create_int_feature([nq_feature.unique_id])
            features["input_ids"] = create_int_feature(nq_feature.input_ids)
            features["input_mask"] = create_int_feature(nq_feature.input_mask)
            features["segment_ids"] = create_int_feature(nq_feature.segment_ids)

            if self.is_training:
                features["start_positions"] = create_int_feature(
                        [nq_feature.start_position])
                features["end_positions"] = create_int_feature(
                        [nq_feature.end_position])
                features["answer_types"] = create_int_feature(
                        [nq_feature.answer_type])
            else:
                token_map = [-1] * len(nq_feature.input_ids)
                for k, v in nq_feature.token_to_orig_map.items():
                    token_map[k] = v
                features["token_map"] = create_int_feature(token_map)

            yield tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()        
        
    def process_nq_lines(self, nq_lines, output_tfrecord, max_examples=0, collect_nq_features=False):

        examples_processed = 0
        instance_processed = 0
        num_examples_with_correct_context = 0

        num_spans_to_ids = dict()

        collected_nq_features = None
        if collect_nq_features:
            collected_nq_features = []

        with tf.io.TFRecordWriter(output_tfrecord) as writer:
            for nq_line in nq_lines:
                nq_entry = nq_line_to_nq_entry(nq_line)
                instances = list(self.process_nq_entry(nq_entry, collected_nq_features))

                if len(instances) not in num_spans_to_ids:
                    num_spans_to_ids[len(instances)] = 0
                num_spans_to_ids[len(instances)] += 1

                for instance in instances:
                    writer.write(instance)
                    instance_processed += 1
                examples_processed += 1
                if nq_entry["has_correct_context"]:
                    num_examples_with_correct_context += 1
                if examples_processed % 10 == 0:
                    print("Examples processed: {}".format(examples_processed))
                    print("Instance processed: {}".format(instance_processed))
                    print("-" * 80)
                if max_examples > 0 and examples_processed >= max_examples:
                    break
            print("Examples with correct context retained: {} of {}".format(num_examples_with_correct_context, examples_processed))
            print("Total number of instances: {}".format(instance_processed))
            if collected_nq_features is not None:
                print("Total number of features: {}".format(len(collected_nq_features)))

            print(json.dumps(num_spans_to_ids, ensure_ascii=False, indent=4))

        return collected_nq_features

    def process_nq_jsonl_files(self, jsonl_files, output_tfrecord, max_examples=0):

        raw_example_iterator = jsonl_iterator(jsonl_files, to_json=False)

        self.process_nq_lines(raw_example_iterator, output_tfrecord=output_tfrecord, max_examples=max_examples)



