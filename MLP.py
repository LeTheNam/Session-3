import tensorflow as tf 
import numpy as np
from LoadData import DataReader

NUM_CLASSES = 20
class MLP:
    def __init__(self, vocab_size, hidden_size):
        self._vocab_size  = vocab_size
        self._hidden_size = hidden_size

    def build_graph(self):
        self._X = tf.placeholder(tf.float32, shape= [None, self._vocab_size])
        self._real_Y = tf.placeholder(tf.int32, shape= [None, ])

        # weights, bias
        
        weight_1 = tf.get_variable(
            name= 'weight_input_hidden',
            shape= (self._vocab_size,self._hidden_size),
            initializer= tf.random_normal_initializer(seed=2020)
        )

        biases_1 = tf.get_variable(
            name= 'biases_input_hidden',
            shape= (self._hidden_size),
            initializer= tf.random_normal_initializer(seed=2020)          
        )

        weight_2 = tf.get_variable(
            name= 'weight_output_hidden',
            shape= (self._hidden_size, NUM_CLASSES),
            initializer= tf.random_normal_initializer(seed=2020)
        )
        biases_2 = tf.get_variable(
            name= 'biases_output_hidden',
            shape= (NUM_CLASSES),
            initializer= tf.random_normal_initializer(seed=2020)
        )

        # Computation graph
        hidden = tf.matmul(self._X, weight_1) + biases_1
        hidden = tf.sigmoid(hidden)
        logits = tf.matmul(hidden, weight_2)+ biases_2

        labels_one_hot = tf.one_hot(indices=self._real_Y, depth= NUM_CLASSES, dtype= tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels= labels_one_hot, logits= logits)
        loss = tf.reduce_mean(loss)

        # lấy predicted-labels để tính accuracy
        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis= 1)
        predicted_labels = tf.squeeze(predicted_labels)

        return predicted_labels, loss

    def trainer(self, loss, learning_rate):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op

# Save parameters of model
def save_parameters(name, value, epoch):
    filename = name.replace(':','-colon-') + '-epoch-{}.txt'.format(epoch)
    if len(value.shape) == 1: # is a list
        string_form = ','.join([str(number) for number in value])
    else:
        string_form = '\n'.join([','.join([str(number) for number in value[row]]) for row in range(value.shape[0])])
    with open( './DS_LAB/Session_3/' + filename, 'w') as  f:
        f.write(string_form)
    return epoch

# Restore parameters
def restore_parameters(name, epoch):
    filename = name.replace(':','-colon-')+ '-epoch-{}.txt'.format(epoch)
    with open('./DS_LAB/Session_3/' + filename) as f:
        lines = f.read().splitlines()
        if len(lines) == 1: # là 1 list
            value = [float(number) for number in lines[0].split(',')]
        else: #is a matrix
            value= [[float(number) for number in lines[row].split(',')] for row in range(len(lines))]
    return value



def main():
    # Create a computation graph
    with open('./DS_LAB/Session_3/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())

    mlp = MLP(vocab_size= vocab_size, hidden_size= 50)
    predicted_labels, loss = mlp.build_graph()
    train_op = mlp.trainer(loss= loss, learning_rate= 0.1)

    def load_dataset():
        train_data_reader = DataReader( data_path='./DS_LAB/Session_3/train_tf_idf.txt', 
                                        batch_size=50, 
                                        vocab_size= vocab_size)
        test_data_reader = DataReader( data_path='./DS_LAB/Session_3/test_tf_idf.txt', 
                                        batch_size=50, 
                                        vocab_size= vocab_size)
        return train_data_reader, test_data_reader
    # mở một phiên chạy đồ thị
    epo = 0
    with tf.Session() as sess:
        train_data_reader, test_data_reader = load_dataset()
        step, MAX_STEP = 0, 5000
        sess.run(tf.global_variables_initializer())
        while step < MAX_STEP:
            train_data, train_labels = train_data_reader.next_batch()
            predicted_label_eval, loss_eval, _ = sess.run(
                [predicted_labels, loss, train_op],
                feed_dict= {
                    mlp._X: train_data,
                    mlp._real_Y: train_labels
                }
            )
            step += 1
            print("step: {}, loss: {}".format(step, loss_eval))
            
        # lưu các tham số của mô hình trong quá trình training
        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            epo = save_parameters(
                    name= variable.name,
                    value= variable.eval(),
                    epoch=train_data_reader._num_epoch
                    )

    # Danh gia model voi test data
    test_data_reader = DataReader(
                            data_path='./DS_LAB/Session_3/test_tf_idf.txt', 
                            batch_size=50, 
                            vocab_size= vocab_size
                            )

    #Create a session
    with tf.Session() as sess:
        epoch = epo

        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            saved_value = restore_parameters(variable.name, epoch)
            assign_op = variable.assign(saved_value)
            sess.run(assign_op)
        
        num_true_pred = 0
        while True:
            test_data, test_labels = test_data_reader.next_batch()
            test_plabel_eval= sess.run(
                predicted_labels,
                feed_dict={
                    mlp._X: test_data,
                    mlp._real_Y: test_labels
                }
            )
            matches = np.equal(test_plabel_eval, test_labels)
            num_true_pred += np.sum(matches.astype(float))

            if test_data_reader._batch_id == 0:
                break

        print("Epoch: ", epoch)
        print("Accuracy on test data: ", num_true_pred/len(test_data_reader._data))

if __name__=="__main__":
    main()