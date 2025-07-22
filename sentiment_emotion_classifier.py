import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # 4GB memory
    except RuntimeError as e:
        print(e)


training_data = pd.read_csv('/home/sohail/emotion/training.csv')
validation_data = pd.read_csv('/home/sohail/emotion/validation.csv')
testing_data = pd.read_csv('/home/sohail/emotion/test.csv')

def preprocess_text(text):
    """
    Preprocesses a single text input using simple tokenization
    """
    
    text = str(text)
    
   
    text = text.lower()
    
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    
    text = ' '.join(text.split())
    
   
    tokens = text.split()
    
    return text, tokens


def process_dataset(df, text_column='text', label_column='label', dataset_name=''):
    """
    Process a single dataset and return statistics
    """
    print(f"\n=== Processing {dataset_name} ===")
    print(f"Initial shape: {df.shape}")
    
   
    initial_size = len(df)
    df = df.dropna(subset=[text_column, label_column])
    print(f"Rows with missing values removed: {initial_size - len(df)}")
    
    
    df['cleaned_text'], df['tokens'] = zip(*df[text_column].apply(preprocess_text))
    
    # Calculate statistics
    avg_token_length = df['tokens'].apply(len).mean()
    print(f"Average tokens per text: {avg_token_length:.2f}")
    
    # Show label distribution
    print("\nLabel distribution:")
    label_dist = df[label_column].value_counts().sort_index()
    total = len(df)
    
    # Updated emotion mapping
    emotion_map = {
        0: 'sadness',
        1: 'joy',
        2: 'love',
        3: 'anger',
        4: 'fear',
        5: 'surprise'  # Added new emotion
    }
    
    for label, count in label_dist.items():
        percentage = (count/total) * 100
        emotion = emotion_map.get(label, f'unknown_{label}')  # Safely handle any unexpected labels
        print(f"{emotion} ({label}): {count} ({percentage:.1f}%)")
    
    # Print unique labels for verification
    print("\nUnique labels in dataset:", sorted(df[label_column].unique()))
    
    return df

processed_training = process_dataset(training_data, dataset_name='Training Data')
processed_validation = process_dataset(validation_data, dataset_name='Validation Data')
processed_testing = process_dataset(testing_data, dataset_name='Testing Data')

# Show sample of processed data
print("\n=== Sample of processed training data ===")
print(processed_training[['cleaned_text', 'tokens']].head(2))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextFeatureEngineering:
    def __init__(self, max_features=5000, max_length=1000):
        self.max_features = max_features
        self.max_length = max_length
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words='english'
        )
        self.tokenizer = Tokenizer(num_words=max_features)
        
    def create_tfidf_features(self, train_texts, val_texts=None, test_texts=None):
        """
        Create TF-IDF features for text data
        """
        print("Creating TF-IDF features...")
        # Fit and transform training data
        X_train_tfidf = self.tfidf.fit_transform(train_texts)
        
        # Transform validation and test if provided
        X_val_tfidf = self.tfidf.transform(val_texts) if val_texts is not None else None
        X_test_tfidf = self.tfidf.transform(test_texts) if test_texts is not None else None
        
        # Get feature names for analysis
        feature_names = self.tfidf.get_feature_names_out()
        print(f"Number of TF-IDF features: {len(feature_names)}")
        
        return {
            'train': X_train_tfidf,
            'val': X_val_tfidf,
            'test': X_test_tfidf,
            'feature_names': feature_names
        }
    
    def create_sequence_features(self, train_texts, val_texts=None, test_texts=None):
        """
        Create sequence features for deep learning models
        """
        print("Creating sequence features...")
        # Fit tokenizer on training data
        self.tokenizer.fit_on_texts(train_texts)
        
        # Convert texts to sequences
        X_train_seq = self.tokenizer.texts_to_sequences(train_texts)
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_length, padding='post')
        
        # Process validation and test if provided
        X_val_pad = None
        if val_texts is not None:
            X_val_seq = self.tokenizer.texts_to_sequences(val_texts)
            X_val_pad = pad_sequences(X_val_seq, maxlen=self.max_length, padding='post')
            
        X_test_pad = None
        if test_texts is not None:
            X_test_seq = self.tokenizer.texts_to_sequences(test_texts)
            X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_length, padding='post')
        
        print(f"Vocabulary size: {len(self.tokenizer.word_index) + 1}")
        print(f"Sequence length: {self.max_length}")
        
        return {
            'train': X_train_pad,
            'val': X_val_pad,
            'test': X_test_pad,
            'tokenizer': self.tokenizer
        }
    
    def analyze_tfidf_features(self, tfidf_features, labels):
        """
        Analyze the most important features for each emotion
        """
        print("\nAnalyzing important features for each emotion...")
        feature_names = tfidf_features['feature_names']
        X_train_tfidf = tfidf_features['train']
        
        for emotion in sorted(set(labels)):
            # Get indices for this emotion
            emotion_indices = labels == emotion
            
            # Calculate average TF-IDF scores for this emotion
            emotion_scores = X_train_tfidf[emotion_indices].mean(axis=0).A1
            
            # Get top features
            top_indices = emotion_scores.argsort()[-10:][::-1]
            top_features = [(feature_names[i], emotion_scores[i]) for i in top_indices]
            
            print(f"\nTop features for emotion {emotion}:")
            for feature, score in top_features:
                print(f"{feature}: {score:.4f}")

def create_features(processed_training, processed_validation, processed_testing):
    """
    Create features for all datasets
    """
    # Initialize feature engineering
    feature_eng = TextFeatureEngineering(max_features=5000, max_length=100)
    
    print("Processing features for training set of shape:", processed_training.shape)
    
    # Create TF-IDF features
    tfidf_features = feature_eng.create_tfidf_features(
        processed_training['cleaned_text'],
        processed_validation['cleaned_text'],
        processed_testing['cleaned_text']
    )
    
    # Create sequence features for deep learning
    sequence_features = feature_eng.create_sequence_features(
        processed_training['cleaned_text'],
        processed_validation['cleaned_text'],
        processed_testing['cleaned_text']
    )
    
    # Analyze features
    feature_eng.analyze_tfidf_features(tfidf_features, processed_training['label'])
    
    return tfidf_features, sequence_features

tfidf_features, sequence_features = create_features(processed_training, processed_validation, processed_testing)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam

max_length = 59  
embedding_dim = 200  
num_classes = 6
max_features = 1000  


def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefficients
    return embeddings_index

def create_embedding_matrix(tokenizer, embeddings_index, embedding_dim):
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def build_lstm_model(embedding_matrix, max_length, num_classes):
    model = Sequential()
    # Embedding Layer
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                        output_dim=embedding_matrix.shape[1],
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=max_length,
                        trainable=False))  # Set trainable=True if you want to fine-tune
    
    # LSTM Layer 1
    model.add(LSTM(512, return_sequences=True,))  # Increased units to 256
    
    # LSTM Layer 2
    model.add(LSTM(256, return_sequences=False,))  # Increased units to 128
    
    # Dense Layer
    model.add(Dense(128, activation='relu'))  # Increased units to 128
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    
    return model


import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(processed_training['cleaned_text'])
X_train_seq = tokenizer.texts_to_sequences(processed_training['cleaned_text'])
X_val_seq = tokenizer.texts_to_sequences(processed_validation['cleaned_text'])
X_test_seq = tokenizer.texts_to_sequences(processed_testing['cleaned_text'])
glove_file = '/home/sohail/glove/glove.6B.200d.txt'

# Pad the sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
embedding_index = load_glove_embeddings(glove_file)  # Update path
embedding_matrix = create_embedding_matrix(tokenizer, embedding_index, embedding_dim)

import pickle

# After fitting the tokenizer on your training data
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Find the maximum sequence length in training, validation, and test sets
max_seq_train = max(len(seq) for seq in X_train_seq)
max_seq_val = max(len(seq) for seq in X_val_seq)
max_seq_test = max(len(seq) for seq in X_test_seq)

# Overall maximum sequence length across all datasets
overall_max_seq = max(max_seq_train, max_seq_val, max_seq_test)

# Print the results
print(f"Max sequence length in training set: {max_seq_train}")
print(f"Max sequence length in validation set: {max_seq_val}")
print(f"Max sequence length in test set: {max_seq_test}")
print(f"Overall max sequence length: {overall_max_seq}")


# Find the maximum sequence length in each set
train_max_length = max(len(seq) for seq in X_train_seq)
val_max_length = max(len(seq) for seq in X_val_seq)
test_max_length = max(len(seq) for seq in X_test_seq)

# Determine the overall maximum sequence length
max_length = max(train_max_length, val_max_length, test_max_length)

print(f"Max sequence length in training set: {train_max_length}")
print(f"Max sequence length in validation set: {val_max_length}")
print(f"Max sequence length in test set: {test_max_length}")
print(f"Overall max sequence length: {max_length}")


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Extract labels from the datasets
y_train = processed_training['label']
y_val = processed_validation['label']
y_test = processed_testing['label']

# Convert labels to numerical values
label_encoder = LabelEncoder()

# Fit the label encoder on the training data and transform all sets
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

# Convert the numerical labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=6)  # Assuming 6 emotion classes
y_val = to_categorical(y_val, num_classes=6)
y_test = to_categorical(y_test, num_classes=6)


model = build_lstm_model(embedding_matrix, max_length, num_classes)


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


history = model.fit(X_train_pad, y_train, 
                    epochs=20, 
                    batch_size=64, 
                    validation_data=(X_val_pad, y_val),
                    )

model.save('sentiment_anaylisis.keras')

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Load your trained model
model = load_model('sentiment_anaylisis.keras')

# Load the tokenizer from file
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the sentiment labels
emotion_labels = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

def predict_sentiment(user_input, tokenizer, max_length):
    # Preprocess input (tokenizing and padding)
    input_seq = tokenizer.texts_to_sequences([user_input])
    input_pad = pad_sequences(input_seq, maxlen=max_length, padding='post')

    # Make a prediction
    prediction = model.predict(input_pad)
    
    # Get the sentiment with the highest probability
    predicted_label = np.argmax(prediction, axis=1)[0]
    
    # Map the predicted label to the sentiment
    predicted_sentiment = emotion_labels[predicted_label]
    
    return predicted_sentiment

# Example usage


user_input = "i absoulty hate you"
predicted_sentiment = predict_sentiment(user_input, tokenizer, max_length=63)
print(f"Predicted sentiment: {predicted_sentiment}")
