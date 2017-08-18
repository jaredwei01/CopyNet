from zutil.config import Config
import tensorflow as tf
import data_util
import random
import seq2seq # from local directory

from tensorflow.python.layers import core as layers_core

config = Config('parameters.json')

# prepare data
vocab = data_util.Vocabulary()
vocab.build_from_file('small.txt')
print 'got %d words' % (vocab.size)
train_data = data_util.Dataset(vocab, config.copy(dataset_filepath='train.txt'))
eval_data = data_util.Dataset(vocab, config.copy(dataset_filepath='eval.txt'))
infer_data = data_util.Dataset(vocab, config.copy(dataset_filepath='infer.txt'))

# placeholder for inputs
encoder_inputs = tf.placeholder(tf.int32, shape=(config.batch_size, None))
decoder_inputs = tf.placeholder(tf.int32, shape=(config.batch_size, None))
decoder_outputs = tf.placeholder(tf.int32, shape=(config.batch_size, None))

# placeholder for sequence lengths
encoder_inputs_lengths = tf.placeholder(tf.int32, shape=(config.batch_size,))
decoder_inputs_lengths = tf.placeholder(tf.int32, shape=(config.batch_size,))


def build_model(mode='train'):
    assert mode in {'train', 'eval', 'infer'}, 'invalid mode!'
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
    attention_mechanism = seq2seq.LuongAttention(
                            config.decoder_hidden_size, 
                            encoder_outputs, 
                            memory_sequence_length=encoder_inputs_lengths)
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(config.decoder_hidden_size)
    decoder_cell = seq2seq.AttentionWrapper(
                        decoder_cell, 
                        attention_mechanism, 
                        attention_layer_size=config.encoder_hidden_size)
    decoder_initial_state = decoder_cell.zero_state(config.batch_size, 
                                                dtype=tf.float32)

    # decoder
    if mode == 'infer':
        decoder_initial_state = decoder_cell.zero_state(config.batch_size, 
                                                    dtype=tf.float32)
        helper = seq2seq.GreedyEmbeddingHelper(
                    embedding_matrix, 
                    tf.fill([config.batch_size], vocab.word2id[u"START"]),
                    vocab.word2id[u"END"])
        projection_layer = layers_core.Dense(vocab.size, use_bias=False)
        decoder = seq2seq.BasicDecoder(decoder_cell,
                                                  helper,
                                                  decoder_initial_state,
                                                  output_layer=projection_layer)
        maximum_iterations = tf.round(tf.reduce_max(encoder_inputs_lengths) * 2)
        final_outputs, final_state, seq_len = \
                seq2seq.dynamic_decode(decoder,
                                    maximum_iterations=maximum_iterations)
        translations = final_outputs.sample_id
        return translations

    # train or eval mode
    helper = seq2seq.CopyNetTrainingHelper(decoder_inputs_emb, encoder_inputs,
                                               decoder_inputs_lengths, 
                                               time_major=False)
    projection_layer = layers_core.Dense(vocab.size, use_bias=False)
    decoder = seq2seq.CopyNetDecoder(decoder_cell,
                                              helper,
                                              decoder_initial_state,
                                              encoder_outputs,
                                              output_layer=projection_layer)
    final_outputs, final_state, seq_lens = \
                        seq2seq.dynamic_decode(decoder)
    logits = final_outputs.rnn_output

    # loss
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=decoder_outputs, logits=logits)
    max_seq_len = logits.shape[1].value
    target_weights = tf.sequence_mask(decoder_inputs_lengths, max_seq_len, 
                                        dtype=tf.float32)
    loss = tf.reduce_sum(crossent * target_weights / tf.to_float(
        config.batch_size))

    if mode == 'eval':
        return loss

    # gradient clip
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                  config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    return loss, train_op


def train():
    with tf.variable_scope('root'):
        train_loss, train_op = build_model(mode='train')

    with tf.variable_scope('root', reuse=True):
        eval_loss = build_model(mode='eval')

    with tf.variable_scope('root', reuse=True):
        translations = build_model(mode='infer')

    # session configure for GPU
    sess_config = tf.ConfigProto()
    # sess_config.gpu_options.allow_growth = True
    # sess_config.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=sess_config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for batchid in range(config.max_batchid):
            # training
            train_batch = train_data.next_batch()
            feed_dict = {
                encoder_inputs: train_batch.encoder_inputs,
                decoder_inputs: train_batch.decoder_inputs,
                decoder_outputs: train_batch.decoder_outputs,
                encoder_inputs_lengths: train_batch.encoder_inputs_lengths,
                decoder_inputs_lengths: train_batch.decoder_inputs_lengths
            }
            loss_train, _ = sess.run([train_loss, train_op], feed_dict=feed_dict)
            print 'batchid = %d, train loss = %f' % (batchid, loss_train)

            # save checkpoints
            if (batchid + 1) % config.save_freq == 0:
                saver.save(sess, './', global_step=batchid)
                print 'model saved'

            # eval
            if (batchid + 1) % config.eval_freq == 0:
                eval_batch = eval_data.next_batch()
                feed_dict = {
                    encoder_inputs: eval_batch.encoder_inputs,
                    decoder_inputs: eval_batch.decoder_inputs,
                    decoder_outputs: eval_batch.decoder_outputs,
                    encoder_inputs_lengths: eval_batch.encoder_inputs_lengths,
                    decoder_inputs_lengths: eval_batch.decoder_inputs_lengths
                }
                loss_eval = sess.run(eval_loss, feed_dict=feed_dict)
                print '\t\tbatchid = %d, eval loss = %f' % (batchid, loss_eval)

            # inference
            if (batchid + 1) % config.infer_freq == 0:
                infer_batch = infer_data.next_batch()
                feed_dict = {
                    encoder_inputs: train_batch.encoder_inputs,
                    decoder_inputs: train_batch.decoder_inputs,
                    decoder_outputs: train_batch.decoder_outputs,
                    encoder_inputs_lengths: train_batch.encoder_inputs_lengths,
                    decoder_inputs_lengths: train_batch.decoder_inputs_lengths
                }
                word_ids = sess.run(translations, feed_dict=feed_dict)
                print 'batchid = %d' % (batchid)
                index = random.randint(0, 39)
                print 'input:', vocab.word_ids_to_sentence(
                        train_batch.encoder_inputs[index]), '\n'
                print 'output:', vocab.word_ids_to_sentence(
                        train_batch.decoder_inputs[index]), '\n'
                print 'predict:', vocab.word_ids_to_sentence(
                        word_ids[index]), '\n'


if __name__ == '__main__':
    train()
