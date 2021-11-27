import tensorflow as tf
import matplotlib.pyplot as plt

# converted into word numbers.
def Dataprocess(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    # Cut the word numbers according to the spaces and put them into a one-dimensional vector.
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # word number string -> int
    dataset = dataset.map(
        lambda string: tf.string_to_number(string, tf.int32))
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset

# filling and batching
def Dataset_construct(src_path, trg_path, batch_size):
    src_data = Dataprocess(src_path)
    trg_data = Dataprocess(trg_path)

    dataset = tf.data.Dataset.zip((src_data, trg_data))

    # Delete sentences with empty content (only containing <EOS>) and sentences that are too long.
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(
            tf.greater(src_len, 1), tf.less_equal(src_len, 50))
        trg_len_ok = tf.logical_and(
            tf.greater(trg_len, 1), tf.less_equal(trg_len, 50))
        return tf.logical_and(src_len_ok, trg_len_ok)
    dataset = dataset.filter(FilterLength)
    

    # generate sentences "<sos> X Y Z"
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))
    dataset = dataset.map(MakeTrgInput)

    # randomly scramble the training data.
    dataset = dataset.shuffle(10000)

    padded_shapes = (
        (tf.TensorShape([None]),      
         tf.TensorShape([])),         
        (tf.TensorShape([None]),      
         tf.TensorShape([None]),     
         tf.TensorShape([])))       
    # call the padded_batch method to perform batching operations.
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset




# define model
class NMTModel(object):
    def __init__(self):
        # Define the LSTM structure used by the encoder and decoder, double-layer LSTM
        self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(1024)
        self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(1024)
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
          [tf.nn.rnn_cell.BasicLSTMCell(1024) 
           for _ in range(2)])

        # Define word vectors 
        self.src_embedding = tf.get_variable(
            "src_emb", [10000, 1024])
        self.trg_embedding = tf.get_variable(
            "trg_emb", [4000, 1024])

        # define softmax layer
        self.softmax_weight = tf.transpose(self.trg_embedding)

        self.softmax_bias = tf.get_variable(
            "softmax_bias", [4000])


    # forward compute
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]
    
        # Convert input and output word numbers into word vectors.
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)
        
        # Perform dropout on the word vector.
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        # use dynamic_rnn to create encoder
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.enc_cell_fw, self.enc_cell_bw, src_emb, src_size, 
                dtype=tf.float32)
            # concatenate the output of two LSTMs into a tensor.
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)     

        with tf.variable_scope("decoder"):
            # Choose the calculation model of the attention weight.
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                1024, enc_outputs,
                memory_sequence_length=src_size)

            # Encapsulate the lstm and attention together 
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.dec_cell, attention_mechanism,
                attention_layer_size=1024)

            # use attention_cell and dynamic_rnn to build encoder
            dec_outputs, _ = tf.nn.dynamic_rnn(
                attention_cell, trg_emb, trg_size, dtype=tf.float32)

        # compute loss
        output = tf.reshape(dec_outputs, [-1, 1024])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(trg_label, [-1]), logits=logits)

        # set the weight of the fill position to 0
        label_weights = tf.sequence_mask(
            trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)
        
        # defining the backpropagation operation
        trainable_variables = tf.trainable_variables()

        # define optimizer
        grads = tf.gradients(cost / tf.to_float(batch_size),
                             trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables))
        return cost_per_token, train_op



cost_l = []

def run_epoch(session, cost_op, train_op, saver, step):
    global cost_l
    # train a epoch, repeat the training steps until all the data in the Dataset have been traversed.
    while True:
        try:
            # compute the cost
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print(" %d steps cost %.3f" % (step, cost))
                cost_l.append(cost)
            if step % 200 == 0:
                saver.save(session, MODEL_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step

def main():
    # define random input
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # define model
    with tf.variable_scope("nmt_model", reuse=None, 
                           initializer=initializer):
        train_model = NMTModel()
  
    # data process
    data = Dataset_construct(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()
 
    # forward compute graph
    cost_op, train_op = train_model.forward(src, src_size, trg_input,
                                            trg_label, trg_size)

    # train the model
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)
    # plot the cost
    plt.plot(cost_l)
    plt.show()


if __name__ == "__main__":

    SRC_TRAIN_DATA = "./train.en"          # Source language input file.
    TRG_TRAIN_DATA = "./train.zh"          # The target language input file.
    MODEL_PATH = "./attention_ckpt"  
    BATCH_SIZE = 100                       # The size of the training data batch. 
    NUM_EPOCH = 5
    KEEP_PROB = 0.8                        # The probability that the node will not be dropped out.
    SOS_ID  = 1 

    main()