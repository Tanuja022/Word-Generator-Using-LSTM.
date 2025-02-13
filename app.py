import streamlit as st 
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# load lstm model
model=load_model('next_word_lstm.h5')

# load the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)


# function to predict next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    # Step 1: Tokenize the input text
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Step 2: Ensure token list is within the max sequence length
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Take the last max_sequence_len-1 tokens
        # Step 3: Pad the sequence to match the model's input shape
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    # Step 4: Predict the next word
    predicted = model.predict(token_list, verbose=0)

    # Step 5: Get the index of the predicted word
    predicted_word_index = np.argmax(predicted, axis=1)[0]  # Get the index of the word with the highest probability

    # Step 6: Reverse the word index to find the corresponding word
    reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}

    # Step 7: Return the predicted word
    predicted_word = reverse_word_index.get(predicted_word_index, None)

    return predicted_word


# streamlit app
st.title("next word prediction with LSTM and EarlyStoppping")
input_text=st.text_input("Enter the sequence of words","Enter Barnardo and Francisco two ")
if st.button("predict next word"):
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next word: {next_word}")
