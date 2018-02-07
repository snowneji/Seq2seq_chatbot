import numpy as np
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dropout,Dense,RepeatVector
from keras.layers.core import Flatten
from keras.engine.topology import Input
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
import gc
import re
import string
from gensim.models import KeyedVectors 
import  xml.dom.minidom
from nltk.tokenize import word_tokenize
import pathlib,glob
import pickle






def text_clean(text):
    """Basic text cleaning."""
    punctuations = string.punctuation.replace("'"," ")
    dat = text
    dat = dat.apply(lambda x: str(x).lower())

    remove = string.punctuation
    pattern = r"[{}]".format(remove)  # create the pattern
    dat = dat.apply(lambda x: re.sub(pattern, '', x))
    dat = dat.apply(lambda x: re.sub(r"[0-9]", "", x))
    dat = dat.apply(lambda i: ''.join(i.strip(punctuations)))
    return(dat)





class Seq2seq(object):
    def __init__(self,
        batch_size,epochs,
        latent_dim,max_words,
        max_len,vl_ratio,tokenizer):

        self.batch_size = batch_size
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.max_words = max_words
        self.max_len = max_len
        self.vl_ratio = vl_ratio
        self.tokenizer = tokenizer
        



    def fit_model (self,
        encoder_input_data,
        decoder_input_data,decoder_target_data):

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.max_words))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        #discard 'encoder_outputs' only keep states.
        encoder_states = [state_h, state_c]


        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.max_words))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.max_words, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # # Compile and run
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')#'categorical_crossentropy')
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_split=self.vl_ratio)
        


        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models

        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        target_token_index = self.tokenizer.word_index
        target_token_index[' '] = 0


        reverse_input_char_index = dict(
            (i, char) for char, i in target_token_index.items())
     
        target_token_index = self.tokenizer.word_index

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.reverse_input_char_index = reverse_input_char_index
        self.target_token_index = target_token_index

        # Save for future query:
        print('Save models and pars for  query:')
        encoder_model.save('encoder_model.h5')
        decoder_model.save('decoder_model.h5')
        # Other pars:
        model_hpars = {
        'max_words':self.max_words,
        'reverse_input_char_index': self.reverse_input_char_index,
        'max_len': self.max_len,
        'tokenizer': self.tokenizer}
        pickle.dump( model_hpars, open( "model_hpars.p", "wb" ) )


    # def decode_sequence(self, input_seq):
    #     # Encode the input as state vectors.
    #     states_value = self.encoder_model.predict(input_seq)
    #     print('target_seq:')
    #     print(len(states_value))
    #     print(len(states_value[0]))
    #     # Generate empty target sequence of length 1.
    #     target_seq = np.zeros((1, 1, self.max_words))
    #     print('target_seq:')
    #     print(target_seq.shape)
    #     # Populate the first character of target sequence with the start character.
    #     # target_seq[0, 0, target_token_index['\t']] = 1.

    #     # Sampling loop for a batch of sequences
    #     # (to simplify, here we assume a batch of size 1).
    #     stop_condition = False
    #     decoded_sentence = ''
    #     n_words = 0
    #     while not stop_condition:
    #         output_tokens, h, c = self.decoder_model.predict(
    #             [target_seq] + states_value)
    #         # print(output_tokens.shape)

    #         # Sample a token
    #         sampled_token_index = np.argmax(output_tokens[0, 0, :])
    #         if (sampled_token_index==0 or sampled_token_index==1) and n_words<=4:
    #         	sampled_token_index = np.argsort(output_tokens[0, 0, :], axis=-1, kind='quicksort', order=None)[-2]


            
    #         # print('---')
    #         sampled_char = self.reverse_input_char_index[sampled_token_index]
            
            
    #         # Exit condition: either hit max length
    #         # or find stop character.
    #         if n_words > self.max_len or sampled_char=='endofsent':
    #             stop_condition = True
    #         else:
    #             decoded_sentence += sampled_char+' '
                

    #         # Update the target sequence (of length 1).
    #         target_seq = np.zeros((1, 1, self.max_words))
    #         target_seq[0, 0, sampled_token_index] = 1.
    #         # Update states
    #         states_value = [h, c]
    #         n_words+=1

    #     return decoded_sentence










if __name__ == "__main__":
    # Key Parameters
    batch_size = 128  # Batch size for training.
    epochs = 128 #28  # Number of epochs to train for. # maybe 14
    latent_dim = 256  # Latent dimensionality of the encoding space.
    max_words = 8000
    n_example = 100000
    max_len = 9
    vl_ratio = 0.1

    





    dialogue_lines = pickle.load( open( "dialogues.p", "rb" ) )
    dialogue_lines = text_clean(pd.Series(dialogue_lines[:n_example*2]))

    dialogue_lines = [i.strip() for i in dialogue_lines]
    dialogue_lines = [line+' endofsent' for line in dialogue_lines]


    ### Process data:
    tokenizer = Tokenizer(num_words = max_words)
    tokenizer.fit_on_texts(dialogue_lines)
    dialogue_lines = tokenizer.texts_to_sequences(dialogue_lines)
    dialogue_lines = pad_sequences(dialogue_lines,maxlen=max_len,padding='post')
    # dialogue_lines = dialogue_lines[:(len(dialogue_lines)-1)] # Just make sure even number



    # Input and Output text manual Processing:
    # Can use Keras for easier processing
    input_texts = []
    target_texts = []

    for i in range(0,len(dialogue_lines),2):
        input_text = dialogue_lines[i]
        target_text = dialogue_lines[i+1]
        input_texts.append(input_text)
        target_texts.append(target_text)

    del input_text;
    del target_text;
    del dialogue_lines;
    gc.collect();


    # a = dict((i, char) for char, i in tokenizer.word_index.items())

    
    # Get the format correct for the seq2seq
    encoder_input_data = np.zeros(
        (len(input_texts), max_len, max_words),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_len, max_words),
        dtype='float32')

    decoder_target_data = np.zeros(
        (len(input_texts), max_len, max_words),
        dtype='float32')


    # Constructing the input and output:
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):

            #i-th sentence
            #t-th word
            #char actual letter
            encoder_input_data[i, t, char] = 1. 
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, char] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, char] = 1.
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
               



    s2smodel = Seq2seq(
        batch_size = batch_size,
        epochs = epochs,
        latent_dim = latent_dim,
        max_words = max_words,
        max_len = max_len,
        vl_ratio = vl_ratio,
        tokenizer = tokenizer
        )

    # 2.9428
    s2smodel.fit_model(encoder_input_data,decoder_input_data,decoder_target_data)


    # # Testing using training dat:
    # for i in range(10):
    #     input_seq = encoder_input_data[i:i+1]
    #     target_seq = decoder_input_data[i:i+1]
    #     decoded_sentence = s2smodel.decode_sequence(input_seq)
    #     decoded_target = s2smodel.decode_sequence(target_seq)
    #     print('-')
    #     print('Input sentence:', ' '.join([ s2smodel.reverse_input_char_index[j] for j in input_texts[i].tolist() if s2smodel.reverse_input_char_index[j]!='endofsent'] ))
    #     print('Target sentence:', ' '.join([ s2smodel.reverse_input_char_index[j] for j in target_texts[i].tolist() if s2smodel.reverse_input_char_index[j]!='endofsent'] ))
    #     print('Decoded sentence:', decoded_sentence)



    # # Manual Testing:
    # query0 = 'i have a dog'
    # query = s2smodel.tokenizer.texts_to_sequences([query0])
    # query2 = np.zeros(
    #     (1, max_len, max_words),
    #     dtype='float32')
    # for t, char in enumerate(query[0]):
    #     query2[:, t, char] = 1. 






    # decoded_sentence = s2smodel.decode_sequence(query2)
    # print('Q: '+query0)
    # print('A: '+decoded_sentence)






