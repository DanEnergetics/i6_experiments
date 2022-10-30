__all__ = ["init_segment_order_shuffle", "rm_segment_order_shuffle"]

import copy

def init_segment_order_shuffle(system, train_corpus="crnn_train", chunk_size=1000):
    system.csp[train_corpus] = copy.deepcopy(system.csp[train_corpus])
    system.csp[train_corpus].corpus_config.segment_order_shuffle = True
    system.csp[train_corpus].corpus_config.segment_order_sort_by_time_length = True
    system.csp[train_corpus].corpus_config.segment_order_sort_by_time_length_chunk_size = chunk_size

def rm_segment_order_shuffle(system, train_corpus="crnn_train"):
    del system.csp[train_corpus].corpus_config.segment_order_shuffle
    del system.csp[train_corpus].corpus_config.segment_order_sort_by_time_length
    del system.csp[train_corpus].corpus_config.segment_order_sort_by_time_length_chunk_size