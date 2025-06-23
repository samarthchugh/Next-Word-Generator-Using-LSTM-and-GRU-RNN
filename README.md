# 📚 Next Word Generator using LSTM and GRU RNN

This project implements a **Next-Word Prediction** system using deep learning models—**LSTM** and **GRU-based Recurrent Neural Networks (RNNs)**. It is trained on a large corpus of text to predict the most probable next word given a sequence of words.

---

## 🔍 Project Objective

The goal is to understand and compare the performance of **LSTM** and **GRU** networks in sequence modeling tasks like next-word prediction. The system can learn patterns in natural language and generate meaningful suggestions based on prior context.

---

## 🛠️ Features

- Trains RNN-based language models using **LSTM** and **GRU**.
- Implements tokenization and sequence padding using **TensorFlow Keras Tokenizer**.
- Includes **text preprocessing pipeline** (lowercasing, punctuation removal, stopword filtering).
- Predicts next probable words based on user input.
- Performance comparison between LSTM and GRU architectures.

---

## 🧠 Models Used

### 1. Long Short-Term Memory (LSTM)
- Effective in capturing long-term dependencies.
- Mitigates vanishing gradient problems in standard RNNs.

### 2. Gated Recurrent Unit (GRU)
- A simpler and faster alternative to LSTM.
- Comparable performance with fewer parameters.

---

## 📁 Project Structure

<pre lang="markdown"> Next-Word-Generator-Using-LSTM-and-GRU-RNN/
│
├── Next_Word_Prediction.ipynb # Notebook with LSTM/GRU implementation
├── preprocessed_data.pkl # Pickled tokenized data
├── input.txt # Corpus for training
└── README.md </pre>


---

## 🧰 Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **Matplotlib / Seaborn**
- **NLTK**
- **Pickle**

---

## ⚙️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/samarthchugh/Next-Word-Generator-Using-LSTM-and-GRU-RNN.git
   cd Next-Word-Generator-Using-LSTM-and-GRU-RNN

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt

3. **Open either Jupyter Notebook or run through Colab**
    - **Next_Word_Prediction.ipynb**

4. **Train and Predict**
    - Input a sentence like **"Artificial Intellingence is".**
    - The model will predict the next word.

# 🙋 Author
### Samarth Chugh
📧[samarthchugh049@gmail.com](samarthchugh049@gmail.com)
[Samarth Chugh](www.linkedin.com/in/-samarthchugh)