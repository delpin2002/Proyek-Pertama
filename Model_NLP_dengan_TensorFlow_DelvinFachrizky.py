# %% [markdown]
# <h3> SUBMISSION: Proyek Pertama : Membuat Model NLP dengan TensorFlow <h3>
# <h4> Delvin Fachrizky

# %%
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# %%
# PENGGUNAAN CALLBACK

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9 and logs.get('val_accuracy')>0.9):
      print("\n accuracy is above > 90%")
      self.model.stop_training=True
callbacks = myCallback()

# %%
# MENYIAPKAN DATASET
dtfrm = pd.read_csv("kaggle_dataset.csv")
dtfrm.head()

# %%
# MELIHAT VALUE DARI KOLOM GENRE
dtfrm['genre'].value_counts()

# %%
# MENYISIHKAN GENRE YANG TIDAK DIGUNAKAN UNTUK KALI INI DAN HANYA MENGGUNAKAN 3 CONTOH GENRE YAITU COMEDY, ACTION, DAN HORROR
dtfrm = dtfrm[~dtfrm['genre'].isin(['drama','thriller','other','adventure','romance', 'sci-fi'])]
dtfrm['genre'].value_counts()

# %%
# MENGHAPUS KARAKTER YANG TIDAK DIGUNAKAN
dtfrm['Text'] = dtfrm['text'].map(lambda x: re.sub(r'\W+', ' ', x))
# MENGHAPUS KOLOM ID DAN TEXT
dtfrm = dtfrm.drop(['id', 'text'], axis=1)
dtfrm.head()

# %%
# MEMBERIKAN LABEL GENRE
genre = pd.get_dummies(dtfrm.genre)
dtfrm_genre = pd.concat([dtfrm, genre], axis=1)
dtfrm_genre = dtfrm_genre.drop(columns='genre')
dtfrm_genre.head()

# %%
# MENKONVERSI TIPE DATA MENJADI STR DAN MEMBUATNYA KE DALAM NUMPY ARRAY
txt = dtfrm_genre['Text'].astype(str)
label = dtfrm_genre[['action', 'horror', 'comedy']].values

# %%
# MEMBUAT SPLIT DATASET SEBESAR 20% (TEST/VALIDATION)
g_train, g_test, l_train, l_test = train_test_split(txt, label, test_size=0.2)

# %%
# PEMODELAN SEQUENTIAL DENGAN EMBEDDING DAN LSTM 
# TOKENIZER
tokenizer = Tokenizer(num_words=5000, oov_token='x')
tokenizer.fit_on_texts(g_train)

seq_train = tokenizer.texts_to_sequences(g_train)
seq_test = tokenizer.texts_to_sequences(g_test)

pad_train = pad_sequences(seq_train, maxlen=200, truncating="post")
pad_test = pad_sequences(seq_test, maxlen=200, truncating="post")

sq_train = tokenizer.texts_to_sequences(g_train)
sq_test = tokenizer.texts_to_sequences(g_test)

print(pad_test.shape)

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=200),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy',)
model.summary()

# %%
# WAKTU YANG DIBUTUHKAN HANYA KURAND DARI 2 MENIT

num_epochs = 30
history = model.fit(pad_train, l_train, epochs=num_epochs, 
                    validation_data=(pad_test, l_test), verbose=2, callbacks=[callbacks])

# %%
# MEMBUAT GRAFIK VISUALISASI DARI MODEL ACCURACY
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
# MEMBUAT GRAFIK VISUALISASI DARI MODEL LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


