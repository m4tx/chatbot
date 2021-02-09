# Data:
# https://www.kaggle.com/kausr25/chatterbotenglish

import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras import preprocessing, utils


def load_data() -> Tuple[List[str], List[str]]:
    dir_name = 'data'
    files_list = os.listdir(f'{dir_name}/')

    questions = []
    answers = []

    for file_name in files_list:
        with open(f'{dir_name}/{file_name}', 'r') as f:
            yaml_file = yaml.safe_load(f)
            conversations = yaml_file['conversations']

            for conversation in conversations:
                question = conversation[0]
                replies = conversation[1:]
                answer = ' '.join(replies)

                questions.append(question)
                answers.append(answer)

    answers = [f'<START> {x} <END>' for x in answers]

    return questions, answers


def make_model_data(tokenizer: preprocessing.text.Tokenizer, questions: List[str], answers: List[str]):
    vocab_size = len(tokenizer.word_index) + 1

    tokenized_questions = tokenizer.texts_to_sequences(questions)
    maxlen_questions = max([len(x) for x in tokenized_questions])
    padded_questions = preprocessing.sequence.pad_sequences(
        tokenized_questions, maxlen=maxlen_questions, padding='post')
    encoder_input_data = np.array(padded_questions)

    tokenized_answers = tokenizer.texts_to_sequences(answers)
    maxlen_answers = max([len(x) for x in tokenized_answers])
    padded_answers = preprocessing.sequence.pad_sequences(
        tokenized_answers, maxlen=maxlen_answers, padding='post')
    decoder_input_data = np.array(padded_answers)

    tokenized_answers = tokenizer.texts_to_sequences(answers)
    tokenized_answers = [x[1:] for x in tokenized_answers]
    padded_answers = preprocessing.sequence.pad_sequences(
        tokenized_answers, maxlen=maxlen_answers, padding='post')
    one_hot_answers = utils.to_categorical(padded_answers, vocab_size)
    decoder_output_data = np.array(one_hot_answers)

    return encoder_input_data, decoder_input_data, decoder_output_data, maxlen_questions, maxlen_answers


class Seq2SeqModel:
    def __init__(self, vocab_size: int, maxlen_questions: int, maxlen_answers: int):
        self.vocab_size = vocab_size
        self.maxlen_questions = maxlen_questions
        self.maxlen_answers = maxlen_answers

        self._create_model()
        self._create_inference_model()

    def _create_model(self):
        self.encoder_inputs = tf.keras.layers.Input(shape=(self.maxlen_questions,))
        self.encoder_embedding = tf.keras.layers.Embedding(self.vocab_size, 300, mask_zero=True)(self.encoder_inputs)
        self.encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(300, return_state=True)(self.encoder_embedding)
        self.encoder_states = [state_h, state_c]

        self.decoder_inputs = tf.keras.layers.Input(shape=(self.maxlen_answers,))
        self.decoder_embedding = tf.keras.layers.Embedding(self.vocab_size, 300, mask_zero=True)(self.decoder_inputs)
        self.decoder_lstm = tf.keras.layers.LSTM(300, return_state=True, return_sequences=True)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_embedding, initial_state=self.encoder_states)
        self.decoder_dense = tf.keras.layers.Dense(self.vocab_size, activation=tf.keras.activations.softmax)
        self.output = self.decoder_dense(self.decoder_outputs)

        self.model = tf.keras.models.Model([self.encoder_inputs, self.decoder_inputs], self.output)
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

    def _create_inference_model(self):
        self.inference_encoder_model = tf.keras.models.Model(self.encoder_inputs, self.encoder_states)

        decoder_state_input_h = tf.keras.layers.Input(shape=(300,))
        decoder_state_input_c = tf.keras.layers.Input(shape=(300,))

        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = self.decoder_lstm(
            self.decoder_embedding, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.inference_decoder_model = tf.keras.models.Model(
            [self.decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def fit(self, encoder_input_data, decoder_input_data, decoder_output_data, batch_size, epochs):
        self.model.fit([encoder_input_data, decoder_input_data], decoder_output_data,
                       batch_size=batch_size, epochs=epochs)

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path)


def str_to_tokens(tokenizer: preprocessing.text.Tokenizer, sentence: str, maxlen: int):
    tokens_list = tokenizer.texts_to_sequences([sentence])[0]
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen, padding='post')


def predict_answer(model: Seq2SeqModel, tokenizer: preprocessing.text.Tokenizer, maxlen_questions: int,
                   maxlen_answers: int, question: str) -> str:
    states_values = model.inference_encoder_model.predict(str_to_tokens(tokenizer, question, maxlen_questions))
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    decode_more = True
    decoded_words = []

    while decode_more:
        dec_outputs, h, c = model.inference_decoder_model.predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                decoded_words.append(word)
                sampled_word = word
                break

        if sampled_word == 'end' or len(decoded_words) > maxlen_answers:
            decode_more = False
            if sampled_word == 'end':
                decoded_words.pop()

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    return ' '.join(decoded_words).capitalize() + '.'


def main():
    questions, answers = load_data()

    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(questions + answers)
    vocab_size = len(tokenizer.word_index) + 1

    encoder_input_data, decoder_input_data, decoder_output_data, maxlen_questions, maxlen_answers = make_model_data(
        tokenizer, questions, answers)

    model = Seq2SeqModel(vocab_size, maxlen_questions, maxlen_answers)
    # model.fit(encoder_input_data , decoder_input_data, decoder_output_data, batch_size=64, epochs=200)
    # model.save_weights('model.h5')
    model.load_weights('model.h5')

    while True:
        question = input('Enter question: ')
        answer = predict_answer(model, tokenizer, maxlen_questions, maxlen_answers, question)
        print(answer)


if __name__ == '__main__':
    main()
