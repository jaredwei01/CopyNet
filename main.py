from zutil.config import Config
import tensorflow as tf
import data_util

from tensorflow.python.layers import core as layers_core

config = Config('parameters.json')

# prepare data
vocab = data_util.Vocabulary()
vocab.build_from_file('small.txt')
dataset = data_util.Dataset(vocab, config)

# placeholder for inputs
encoder_inputs = tf.placeholder(tf.int32, shape=(config.batch_size, None))
decoder_inputs = tf.placeholder(tf.int32, shape=(config.batch_size, None))
decoder_outputs = tf.placeholder(tf.int32, shape=(config.batch_size, None))

# placeholder for sequence lengths
encoder_inputs_lengths = tf.placeholder(tf.int32, shape=(config.batch_size,))
decoder_inputs_lengths = tf.placeholder(tf.int32, shape=(config.batch_size,))


def build_model():
    # embedding maxtrix
    embedding_matrix = tf.get_variable("embedding_matrix", [
        vocab.size, config.embedding_size])
    encoder_inputs_emb = tf.nn.embedding_lookup(embedding_matrix, encoder_inputs)
    decoder_inputs_emb = tf.nn.embedding_lookup(embedding_matrix, decoder_inputs)

    # encoder
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(config.encoder_hidden_size)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                                encoder_cell,
                                encoder_inputs_emb,
                                sequence_length=encoder_inputs_lengths,
                                time_major=False,
                                dtype=tf.float32)

    # attention wrapper for decoder_cell
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                            config.decoder_hidden_size, 
                            encoder_outputs, 
                            memory_sequence_length=encoder_inputs_lengths)
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(config.decoder_hidden_size)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                        decoder_cell, 
                        attention_mechanism, 
                        attention_layer_size=config.encoder_hidden_size)

    # decoder
    decoder_initial_state = decoder_cell.zero_state(config.batch_size, 
                                                    dtype=tf.float32)
    helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_emb, 
                                               decoder_inputs_lengths, 
                                               time_major=False)
    projection_layer = layers_core.Dense(vocab.size, use_bias=False)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                              helper,
                                              decoder_initial_state,
                                              output_layer=projection_layer)
    final_outputs, final_state, seq_lens = \
                        tf.contrib.seq2seq.dynamic_decode(decoder)
    logits = final_outputs.rnn_output

    # loss
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=decoder_outputs, logits=logits)
    max_seq_len = logits.shape[1].value
    target_weights = tf.sequence_mask(decoder_inputs_lengths, max_seq_len, 
                                        dtype=tf.float32)
    loss = tf.reduce_sum(crossent * target_weights / tf.to_float(
        config.batch_size))

    # gradient clip
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                  config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    return loss, train_op


def train():
    fetches = build_model()

    # session configure for GPU
    sess_config = tf.ConfigProto()
    # sess_config.gpu_options.allow_growth = True
    # sess_config.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(config.max_epoch):
            batch = dataset.next_batch()
            feed_dict = {
                encoder_inputs: batch.encoder_inputs,
                decoder_inputs: batch.decoder_inputs,
                decoder_outputs: batch.decoder_outputs,
                encoder_inputs_lengths: batch.encoder_inputs_lengths,
                decoder_inputs_lengths: batch.decoder_inputs_lengths
            }
            loss, _ = sess.run(fetches, feed_dict=feed_dict)
            print 'epoch = %d, loss = %f' % (epoch, loss)


if __name__ == '__main__':
    train()
