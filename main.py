from zutil.config import Config
import tensorflow as tf
import data_util

config = Config('parameters.json')

# prepare data
vocab = data_util.Vocabulary()
vocab.build_from_file('small.txt')
dataset = data_util.Dataset(vocab, config)

# placeholder for inputs
encoder_inputs = tf.placeholder(tf.int32, shape=(config.batch_size,
                                                 config.encoder_max_seq_len))
decoder_inputs = tf.placeholder(tf.int32, shape=(config.batch_size,
                                                 config.decoder_max_seq_len))
decoder_outputs = tf.placeholder(tf.int32, shape=(config.batch_size,
                                                  config.decoder_max_seq_len))

# placeholder for sequence lengths
encoder_inputs_lengths = tf.placeholder(tf.int32, shape=(config.batch_size,))
decoder_inputs_lengths = tf.placeholder(tf.int32, shape=(config.batch_size,))


def build_model():
    # embedding maxtrix
    embedding_matrix = tf.get_variable("embedding_matrix", [
        vocab.size, config.embedding_size])
    encoder_inputs_emb = tf.nn.embedding_lookup(embedding_matrix,
                                                encoder_inputs)
    decoder_inputs_emb = tf.nn.embedding_lookup(embedding_matrix,
                                                decoder_inputs)

    # encoder
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(config.encoder_hidden_size)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell,
            encoder_inputs_emb,
            sequence_length=encoder_inputs_lengths,
            time_major=False,
            dtype=tf.float32)

    # decoder
    helper = tf.contrib.seq2seq.TrainingHelper(
        decoder_inputs_emb, decoder_inputs_lengths, time_major=False)
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(config.decoder_hidden_size)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                              helper, encoder_state)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
    logits = outputs.rnn_output
    print 'logits = ', logits

    # loss
    # total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #        labels=decoder_outputs, logits=logits)
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        decoder_outputs,
        tf.ones([config.batch_size, config.decoder_max_seq_len]),
    )

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
            loss, _ = sess.run(fetches, feed_dict={
                encoder_inputs: batch.encoder_inputs,
                decoder_inputs: batch.decoder_inputs,
                decoder_outputs: batch.decoder_outputs,
                encoder_inputs_lengths: batch.encoder_inputs_lengths,
                decoder_inputs_lengths: batch.decoder_inputs_lengths
            })
            print 'loss = ', loss


if __name__ == '__main__':
    train()
