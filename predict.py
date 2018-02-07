import numpy as np

def decode_sequence(
    input_seq,
    encoder_model,
    decoder_model,
    max_words,
    max_len,
    reverse_input_char_index):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, max_words))
    # Populate the first character of target sequence with the start character.
    # target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    n_words = 0
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # print(output_tokens.shape)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        if (sampled_token_index==0 or sampled_token_index==1) and n_words<=4:
        	sampled_token_index = np.argsort(output_tokens[0, 0, :], axis=-1, kind='quicksort', order=None)[-2]


        
        # print('---')
        sampled_char = reverse_input_char_index[sampled_token_index]
        
        
        # Exit condition: either hit max length
        # or find stop character.
        if n_words > max_len or sampled_char=='endofsent':
            stop_condition = True
        else:
            decoded_sentence += sampled_char+' '
            

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, max_words))
        target_seq[0, 0, sampled_token_index] = 1.
        # Update states
        states_value = [h, c]
        n_words+=1

    return decoded_sentence