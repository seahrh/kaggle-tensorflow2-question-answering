import absl
import tensorflow as tf
import json
import zipfile

from nq_flags import DEFAULT_FLAGS as FLAGS
from nq_dataset_utils import *
from text_utils import simplify_nq_example

flags = absl.flags

# ----------------------------------------------------------------------------------------
# Used in notebook

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_string("vocab_file", "vocab-nq.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_integer(
    "max_seq_length_for_training", 512,
    "The maximum total input sequence length after WordPiece tokenization for training examples. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_float(
    "include_unknowns_for_training", 0.02,
    "If positive, for converting training dataset, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_boolean(
    "skip_nested_contexts", True,
    "Completely ignore context that are not top level nodes in the page.")

# Used in nq_dataset_utils
flags.DEFINE_integer("max_contexts", 48,
                     "Maximum number of contexts to output for an example.")

flags.DEFINE_integer(
    "max_position", 50,
    "Maximum context position for which to generate special tokens.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_string("train_tf_record", "nq_train.tf_records",
                    "Precomputed tf records for training dataset.")

flags.DEFINE_string("valid_tf_record", "nq_valid.tf_records",
                    "Precomputed tf records for validation dataset.")

flags.DEFINE_string("valid_small_tf_record", "nq_valid_small.tf_records",
                    "Precomputed tf records for a smaller validation dataset.")

flags.DEFINE_string("valid_tf_record_with_labels", "nq_valid_with_labels.tf_records",
                    "Precomputed tf records for validation dataset with labels.")

flags.DEFINE_string("valid_small_tf_record_with_labels", "nq_valid_small_with_labels.tf_records",
                    "Precomputed tf records for a smaller validation dataset with labels.")

# This file should be generated when the notebook is running using the provided test dataset!
flags.DEFINE_string("test_tf_record", "nq_test.tf_records",
                    "Precomputed tf records for test dataset.")

flags.DEFINE_integer(
    "max_examples", 0,
    "If positive, stop once these many training examples have been converted.")

flags.DEFINE_integer(
    "max_validation_examples", 1000,
    "If positive, stop once these many validation examples have been converted.")

# ----------------------------------------------------------------------------------------

# Make the default flags as parsed flags
FLAGS.mark_as_parsed()


def jsonl_iterator(jsonl_files, to_json=False):

    for file_path in jsonl_files:
        with open(file_path, "r", encoding="UTF-8") as fp:
            for jsonl in fp:
                raw_example = jsonl
                if to_json:
                    raw_example = json.loads(jsonl)
                yield raw_example


if __name__ == "__main__":
    """
    DON"T RUN THIS SCRIPT IN KAGGLE KERNEL
    """

    # Simplify and convert validation examples to tf records.
    nq_examples = jsonl_iterator(["v1.0-simplified_nq-dev-all.jsonl"], to_json=True)
    with open("simplified-nq-dev.jsonl", "w", encoding="UTF-8") as fp:
        for nq_example in nq_examples:
            simplified_nq_example = simplify_nq_example(nq_example)
            nq_line = json.dumps(simplified_nq_example, ensure_ascii=False)
            fp.write(nq_line + "\n")

    creator = TFExampleCreator(is_training=False)

    nq_lines = jsonl_iterator(["simplified-nq-dev.jsonl"])
    collected_nq_features = creator.process_nq_lines(nq_lines=nq_lines, output_tfrecord=FLAGS.valid_tf_record, max_examples=0, collect_nq_features=True)

    nq_lines = jsonl_iterator(["simplified-nq-dev.jsonl"])
    nb_examples = 0
    with open("simplified-nq-dev-small.jsonl", "w", encoding="UTF-8") as fp:
        for nq_line in nq_lines:
            fp.write(nq_line)
            nb_examples += 1
            if nb_examples >= FLAGS.max_validation_examples:
                break

    nq_lines = jsonl_iterator(["simplified-nq-dev-small.jsonl"])
    collected_nq_features = creator.process_nq_lines(nq_lines=nq_lines, output_tfrecord=FLAGS.valid_small_tf_record, max_examples=FLAGS.max_validation_examples, collect_nq_features=True)

    # Convert test examples to tf records.
    nq_lines = jsonl_iterator(["simplified-nq-test.jsonl"])
    collected_nq_features = creator.process_nq_lines(nq_lines=nq_lines, output_tfrecord=FLAGS.test_tf_record, max_examples=0, collect_nq_features=True)

    # Convert valid examples with labels to tf records.
    creator = TFExampleCreator(is_training=True)

    nq_lines = jsonl_iterator(["simplified-nq-dev.jsonl"])
    collected_nq_features = creator.process_nq_lines(nq_lines=nq_lines, output_tfrecord=FLAGS.valid_tf_record_with_labels, max_examples=0, collect_nq_features=True)

    nq_lines = jsonl_iterator(["simplified-nq-dev-small.jsonl"])
    collected_nq_features = creator.process_nq_lines(nq_lines=nq_lines, output_tfrecord=FLAGS.valid_small_tf_record_with_labels, max_examples=FLAGS.max_validation_examples, collect_nq_features=True)

    # # RESET some flags to flags specific to training
    # FLAGS.include_unknowns = FLAGS.include_unknowns_for_training
    # FLAGS.max_seq_length = FLAGS.max_seq_length_for_training

    # # Convert training examples (up to FLAGS.max_examples) to tf records.
    # creator = TFExampleCreator(is_training=True)

    # def train_jsonl_iterator():
    #     with zipfile.ZipFile("tensorflow2-question-answering.zip") as zfile:
    #         for finfo in zfile.infolist():
    #             if finfo.filename == "simplified-nq-train.jsonl":
    #                 with zfile.open(finfo, "r") as ifile:
    #                     for line in ifile:
    #                         line = line.decode("utf-8")
    #                         yield line

    # nq_lines = train_jsonl_iterator()
    # creator.process_nq_lines(nq_lines=nq_lines, output_tfrecord=FLAGS.train_tf_record, max_examples=FLAGS.max_examples)
