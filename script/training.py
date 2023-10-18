import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import RandomUniform
from sklearn.metrics import classification_report, f1_score
from script.utils_function import *


class TRAINING_CNN_BLSTM(object):

    def __init__(self, EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, LEARNING_RATE, OPTIMIZER,
                 MAX_LEN, DATA_TRAIN, DATA_TEST):

        self.epochs = EPOCHS
        self.dropout = DROPOUT
        self.dropout_recurrent = DROPOUT_RECURRENT
        self.lstm_state_size = LSTM_STATE_SIZE
        self.conv_size = CONV_SIZE
        self.learning_rate = LEARNING_RATE
        self.optimizer = OPTIMIZER
        self.max_len = MAX_LEN
        self.train_sentence = DATA_TRAIN['input']
        self.test_sentence = DATA_TEST['input']

    # Creating word level and character level embeddings
    def embedding(self):
        labelSet = set()
        words = {}

        for dataset in [self.train_sentence, self.test_sentence]:
            for sentence in dataset:
                for token, char, label in sentence:
                    labelSet.add(label)
                    words[token.lower()] = True

        labelSet = sorted(labelSet)
        self.label2Idx = {}
        for label in labelSet:
            self.label2Idx[label] = len(self.label2Idx)

        case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                    'contains_digit': 6, 'PADDING_TOKEN': 7}
        self.caseEmbeddings = np.identity(len(case2Idx), dtype='float32')

        # Reading GLoVE word embeddings
        self.word2Idx = {}
        self.wordEmbeddings = []

        fEmbeddings = open("data/glove.6B.50d.txt", encoding="utf-8")
        # loop through each word in embeddings
        for idx, line in enumerate(fEmbeddings):
            split = line.strip().split(" ")
            word = split[0]  # embedding word entry

            if len(self.word2Idx) == 0:  # add padding+unknown
                self.word2Idx["PADDING_TOKEN"] = len(self.word2Idx)
                vector = np.zeros(len(split) - 1)  # Zero vector for 'PADDING' word
                self.wordEmbeddings.append(vector)

                self.word2Idx["UNKNOWN_TOKEN"] = len(self.word2Idx)
                vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
                self.wordEmbeddings.append(vector)

            if split[0].lower() in words:
                vector = np.array([float(num) for num in split[1:]])
                self.wordEmbeddings.append(vector)  # word embedding vector
                self.word2Idx[split[0]] = len(self.word2Idx)  # corresponding word dict

        self.wordEmbeddings = np.array(self.wordEmbeddings)

        # dictionary of all possible characters
        self.char2Idx = {"PADDING": 0, "UNKNOWN": 1}
        for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>":
            self.char2Idx[c] = len(self.char2Idx)

        # format: [[wordindices], [caseindices], [padded word indices], [label indices]]
        self.train_set = padding(
            createMatrices(self.train_sentence, self.word2Idx, self.label2Idx, case2Idx, self.char2Idx))
        self.test_set = padding(
            createMatrices(self.test_sentence, self.word2Idx, self.label2Idx, case2Idx, self.char2Idx))

        self.idx2Label = {v: k for k, v in self.label2Idx.items()}

    # Function to create batches
    def createBatches(self):
        self.train_batch, self.train_batch_len = createBatches(self.train_set)
        self.test_batch, self.test_batch_len = createBatches(self.test_set)

    # Model layers
    def build_model(self, print_summary=True, save_model_image=True):

        # character input
        character_input = Input(shape=(None, self.max_len,), name="Character_input")
        embed_char_out = TimeDistributed(
            Embedding(len(self.char2Idx), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
            name="Character_embedding")(
            character_input)

        dropout = Dropout(self.dropout)(embed_char_out)

        # CNN
        conv1d_out = TimeDistributed(
            Conv1D(kernel_size=self.conv_size, filters=30, padding='same', activation='tanh', strides=1),
            name="Convolution")(dropout)
        maxpool_out = TimeDistributed(MaxPooling1D(self.max_len), name="Maxpool")(conv1d_out)
        char = TimeDistributed(Flatten(), name="Flatten")(maxpool_out)
        char = Dropout(self.dropout)(char)

        # word-level input
        words_input = Input(shape=(None,), dtype='int32', name='words_input')
        words = Embedding(input_dim=self.wordEmbeddings.shape[0], output_dim=self.wordEmbeddings.shape[1],
                          weights=[self.wordEmbeddings],
                          trainable=False)(words_input)

        # case-info input
        casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
        casing = Embedding(output_dim=self.caseEmbeddings.shape[1], input_dim=self.caseEmbeddings.shape[0],
                           weights=[self.caseEmbeddings],
                           trainable=False)(casing_input)
        # concat & BLSTM
        output = concatenate([words, casing, char])
        output = Bidirectional(LSTM(self.lstm_state_size,
                                    return_sequences=True,
                                    dropout=self.dropout,  # on input to each LSTM block
                                    recurrent_dropout=self.dropout_recurrent  # on recurrent input signal
                                    ), name="BLSTM")(output)
        output = TimeDistributed(Dense(len(self.label2Idx), activation='softmax'), name="Softmax_layer")(output)

        # set up model
        self.model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer)

        self.init_weights = self.model.get_weights()

        if save_model_image:
            plot_model(self.model, to_file='model_cnn_bilstm.png')
            print('Model is built. Image of model is saved completely')
        if print_summary:
            print(self.model.summary())

    def tag_dataset(self, dataset, model):
        """Tag data with numerical values"""
        correctLabels = []
        predLabels = []
        for i, data in enumerate(dataset):
            tokens, casing, char, labels = data
            tokens = np.asarray([tokens])
            casing = np.asarray([casing])
            char = np.asarray([char])
            pred = model.predict([tokens, casing, char], verbose=False)[0]
            pred = pred.argmax(axis=-1)  # Predict the classes
            correctLabels.append(labels)
            predLabels.append(pred)
        return predLabels, correctLabels

    def train_model(self, name):
        self.f1_test_history = []
        test_batch_copy = self.test_batch.copy()

        for epoch in range(self.epochs):

            for i, batch in enumerate(iterate_minibatches(self.train_batch, self.train_batch_len)):
                labels, tokens, casing, char = batch
                self.model.train_on_batch([tokens, casing, char], labels)

            if epoch % 10 == 0:
                print("Epoch {}/{}".format(epoch, self.epochs))
                np.random.shuffle(test_batch_copy)
                sample_1000 = test_batch_copy[:1000]

                predLabels, correctLabels = self.tag_dataset(sample_1000, self.model)
                f1 = f1_score(np.concatenate(predLabels), np.concatenate(correctLabels), average='macro')
                print(f'Macro F1 Score: {f1} - Testing by sampling 1000 sentences randomly from test data.')

        name_model = name + '.h5'
        self.model.save(name_model)
        print('-' * 60)
        print("Training finished.")
        print(f"Model is saved as {name_model} .")

    def evaluate_model(self):
        print(f'Model is evaluated on test set which has {len(self.test_batch)} sentences.')
        print('-' * 60)
        self.correctLabels = []
        self.predLabels = []

        for i, data in enumerate(self.test_batch):
            tokens, casing, char, labels = data
            tokens, casing, char = np.asarray([tokens]), np.asarray([casing]), np.asarray([char])

            pred = self.model.predict([tokens, casing, char], verbose=False)[0]
            pred = pred.argmax(axis=-1)
            if i % 1500 == 0:
                print(f'Predicting in progress ___ {len(self.test_batch) - i} sentences remaining.')
            self.correctLabels.append(labels)
            self.predLabels.append(pred)
        print('-' * 60)
        report = classification_report(np.concatenate(self.correctLabels), np.concatenate(self.predLabels), target_names=list(self.label2Idx.keys()), zero_division=0)
        print(report)
        