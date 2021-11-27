import sys
from translate import Ui_ENtoZH
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow

import tensorflow as tf
import codecs

import time

# Create a window for the translator
class MainWindow(QWidget, Ui_ENtoZH):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = ''
        self.setupUi(self)
        self.pushButton.clicked.connect(self.translate_process)

    def translate_process(self):

        tf.reset_default_graph()
        self.textBrowser.clear()
        self.textBrowser_2.clear()
        # Define the recurrent neural network model for training
        with tf.variable_scope("nmt_model", reuse=None):
            model = NMTModel()
        text = self.lineEdit.text() + ' <eos>'
        
        with codecs.open(SRC_VOCAB, "r", "utf-8") as f_vocab:
            src_vocab = [w.strip() for w in f_vocab.readlines()]
            src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
        test_en_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
                    for token in text.split()]

        start = time.clock()
        output_op = model.inference(test_en_ids)
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)

        output_ids = sess.run(output_op)
        output_id_process = output_ids[1:-1]
        end = time.clock()
        #calculate the used time of translation process 
        t=end-start

        with codecs.open(TRG_VOCAB, "r", "utf-8") as f_vocab:
            trg_vocab = [w.strip() for w in f_vocab.readlines()]
        output_text = ''.join([trg_vocab[x] for x in output_id_process])

        self.textBrowser.setText(output_text.encode('utf8').decode(sys.stdout.encoding))
        self.textBrowser_2.setText(str(round(t,2)) + 's')
        sess.close()



# define model
class NMTModel(object):
    def __init__(self):
        # define LSTM structure
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

    def inference(self, src_input):

        # Sort the input sentences into batches of size 1.
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        # define encoder
        with tf.variable_scope("encoder"):

            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.enc_cell_fw, self.enc_cell_bw, src_emb, src_size, 
                dtype=tf.float32)

            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)    
        
        # define decoder
        with tf.variable_scope("decoder"):
            # Define the attention mechanism used by the decoder
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                1024, enc_outputs,
                memory_sequence_length=src_size)

            # Encapsulate the decoder's recurrent neural network self.dec_cell and attention together into a higher-level recurrent neural network.
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.dec_cell, attention_mechanism,
                attention_layer_size=1024)
   
        # Set the maximum number of decoding steps.
        MAX_DEC_LEN=100

        with tf.variable_scope("decoder/rnn/attention_wrapper"):

            init_array = tf.TensorArray(dtype=tf.int32, size=0,
                dynamic_size=True, clear_after_read=False)
            # fill in the first word <sos> as the input of the decoder。
            init_array = init_array.write(0, SOS_ID)

            # call attention_cell.zero_state to build the initial loop state.
            init_loop_var = (
                attention_cell.zero_state(batch_size=1, dtype=tf.float32),
                init_array, 0)

            # loop until the decoder outputs <eos> or reaches the maximum number of steps.
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), EOS_ID),
                    tf.less(step, MAX_DEC_LEN-1)))

            def loop_body(state, trg_ids, step):
                # cyclic calculation
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,
                                                 trg_input)
                # forward
                dec_outputs, next_state = attention_cell.call(
                    state=state, inputs=trg_emb)

                # Calculate the logit corresponding to each possible output word, and select the word with the largest logit value as the output of this step.
                output = tf.reshape(dec_outputs, [-1, 1024])
                logits = (tf.matmul(output, self.softmax_weight)
                          + self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)

                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1

            # output result
            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()




if __name__ == "__main__":
    
    MODEL_PATH = "./attention_ckpt-9000"       
    SRC_VOCAB = "./en.vocab"
    TRG_VOCAB = "./zh.vocab"
    SOS_ID = 1
    EOS_ID = 2
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())