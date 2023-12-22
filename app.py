
import os

from flask import Flask, render_template
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D

# pylint: disable=C0103
app = Flask(__name__)


@app.route('/', methods=['POST'])
def hello():
    gejala = request.form.get('gejala')

    # Data
    data = {
    'gejala': [
        'tidak bisa dinyalakan',
        'mesin mengeluarakan bau',
        'bunyi mesin aneh',
        'bergetar tidak normal',
        'motor panas berlebihan',
        'mengeluarkan bunyi berdecit',
        'rem tidak berfungsi',
        'motor tersendat sendat',
        'oli mengalami kebocoran',
        'overheat pada mesin',
        'warna asap putih',
        'bunyi mesin aneh',
        'motor berbunyi knocking',
        'turunnya performa mesin',
        'getaran tidak normal',
        'keluar bercak oli',
        'pengurangan level oli',
        'kebocoran blok mesin',
        'mesin tidak befungsi',
        'kondisi aki melemah',
        'lampu indikator menyala',
        'mesin sulit dihidupkan',
        'stater berbunyi aneh',
        'mesin mati mendadak',
        'mesin bergetar kuat',
        'getaran pada kemudi',
        'berisik saat berkendara',
        'warna asap hitam',
        'bau asap aneh',
        'mesin tidak berdaya',
        'berbunyi saat mengerem',
        'rem terasa licin',
        'kecepatan rem melemah',
        'terdapat gesekan gigi',
        'sulit memindahkan gigi',
        'transmisi berbunyi aneh',
        # Tambahkan gejala lain sesuai kebutuhan
    ],
    'kerusakan': [
        'Mesin',
        'Mesin',
        'Mesin',
        'Kelistrikan',
        'Kelistrikan',
        'Kelistrikan',
        'Sistem Rem',
        'Sistem Rem',
        'Sistem Rem',
        'Overheat',
        'Overheat',
        'Overheat',
        'bunyi ketukan pada mesin',
        'bunyi ketukan pada mesin',
        'bunyi ketukan pada mesin',
        'kebocoran minyak',
        'kebocoran minyak',
        'kebocoran minyak',
        'kegagalan aki',
        'kegagalan aki',
        'kegagalan aki',
        'masalah saat menyala',
        'masalah saat menyala',
        'masalah saat menyala',
        'getaran berlebihan',
        'getaran berlebihan',
        'getaran berlebihan',
        'asap dari knalpot',
        'asap dari knalpot',
        'asap dari knalpot',
        'masalah pengereman',
        'masalah pengereman',
        'masalah pengereman',
        'masalah transmisi',
        'masalah transmisi',
        'masalah transmisi',
        # Tambahkan label lain sesuai kebutuhan
    ]
}

    df = pd.DataFrame(data)
    df.head()

    # Membagi data menjadi training dan testing sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # Tokenisasi teks
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data['gejala'])
    vocab_size = len(tokenizer.word_index) + 1

    # Mengonversi teks menjadi sequences
    X_train = tokenizer.texts_to_sequences(train_data['gejala'])
    X_test = tokenizer.texts_to_sequences(test_data['gejala'])

    # Padding sequences agar memiliki panjang yang sama
    maxlen = max(len(seq) for seq in X_train + X_test)
    X_train_padded = pad_sequences(X_train, padding='pre', maxlen=maxlen) #NOTE
    X_test_padded = pad_sequences(X_test, padding='pre', maxlen=maxlen)

    # Label encoding untuk kolom target
    label_to_index = {label: i for i, label in enumerate(df['kerusakan'].unique())}
    y_train = train_data['kerusakan'].map(label_to_index)
    y_test = test_data['kerusakan'].map(label_to_index)

    # Membuat model
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=maxlen))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(len(label_to_index), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Melatih model
    model.fit(X_train_padded, y_train, epochs=50, validation_data=(X_test_padded, y_test), batch_size=32)

    def text_to_sequence(text):
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=X_train_padded.shape[1])
        return padded_sequence

    # Mengonversi teks menjadi sequence
    input_text = ' '.join(gejala.split(';'))
    input_sequence = text_to_sequence(input_text)

    # Melakukan prediksi
    prediction = model.predict(input_sequence)

    # Mengambil indeks label dengan probabilitas tertinggi
    predicted_category_index = prediction.argmax(axis=1)[0]

    # Mendapatkan nama kategori berdasarkan indeks
    predicted_category = list(label_to_index.keys())[
        list(label_to_index.values()).index(predicted_category_index)]

    # Menampilkan hasil prediksi
    message = f"Kemungkinan Kerusakan: {predicted_category}"

    return jsonify(data=message, status=200)


if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)
    # server_port = os.environ.get('PORT', '8080')
    # app.run(debug=False, port=server_port, host='0.0.0.0')
