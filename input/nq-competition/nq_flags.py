import absl
import tensorflow as tf

flags = absl.flags


# Delete all flags before declare
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


DEFAULT_FLAGS = flags.FLAGS
