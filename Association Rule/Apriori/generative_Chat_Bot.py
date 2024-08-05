import tensorflow as tf
import numpy as np

# Create a simple dataset (you'd replace this with your own dataset)
text = "Hello, how are you doing today?"

# Create a vocabulary of characters
vocab = sorted(set(text))
char_to_index = {char: index for index, char in enumerate(vocab)}
index_to_char = np.array(vocab)

# Convert text to numerical representation
text_as_int = np.array([char_to_index[c] for c in text])

# Create input-output pairs for training
seq_length = 10
X = []
y = []
for i in range(0, len(text_as_int) - seq_length, 1):
    X.append(text_as_int[i:i+seq_length])
    y.append(text_as_int[i+seq_length])

X = np.array(X)
y = np.array(y)

# Define the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(vocab), 256, batch_input_shape=[1, None]),
    tf.keras.layers.GRU(512, return_sequences=True, stateful=True),
    tf.keras.layers.Dense(len(vocab))
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Train the model
for epoch in range(30):
    model.reset_states()
    for i in range(0, len(X)):
        X_batch = np.expand_dims(X[i], axis=0)
        y_batch = np.expand_dims(y[i], axis=0)
        model.fit(X_batch, y_batch, epochs=1, verbose=0)
        if i % 100 == 0:
            print(f"Epoch {epoch+1}, Iteration {i}, Loss: {model.history.history['loss'][0]}")

# Generate text
start_seed = "Hello, how"
num_generate = 50
input_eval = [char_to_index[s] for s in start_seed]
input_eval = tf.expand_dims(input_eval, 0)
generated_text = start_seed

for _ in range(num_generate):
    predictions = model(input_eval)
    predicted_index = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    input_eval = tf.expand_dims([predicted_index], 0)
    generated_text += index_to_char[predicted_index]

print(generated_text)
