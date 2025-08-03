AUFGABE = """
 1. Bag of Words:
     vokabular = [python, ich, wie, geht, liebe, dir, es]
     satz1 = ich liebe python
     satz2 = hallo wie geht es dir denn
     bow1 = [1, 1, 0, 0, 1, 0, 0]
     bow2 = [0, 0, 1, 1, 0, 1, 1]
     Aufgabe:
         Schreibe ein Programm, das für ein vorgegebenes Vokabular
         und einen vorgegebenen Satz das BoW erstellt.
         (Für diejenigen, die Funktionen kennen):
             Schreibe eine Funktion, die ein BoW erstellt
 3. Wende Schritt 1 und Schritt 2 an um den Datensatz ChatbotTraining.csv
    zu analysieren.
 4. Trainiere jetzt den ML Algorithmus mit dem gesamten Datensatz ChatbotTraining.csv
    und teste ihn, indem du einen Satz im Terminal eingibst,dieser in
    ein BoW verwandelt wird und anschließend dieser von der
    predict()-Methode klassifiziert wird und die Klasse ausgibt.
 5. Wie Schritt 4, aber anstatt dass die Kategorie am Ende ausgegeben wird, 
    wird die Information genutzt um eine Antwort aus der response-Spalte mit
    passender Kategorie zufällig auszuwählen.  
"""

import csv
import re
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle


# ---------- Textverarbeitung ----------
def text_to_words(text):
    return set(re.findall(r"\w+|[^\s\w]", text.lower()))

def text_to_vector(text, vocabulary):
    words = text_to_words(text)
    return [1 if word in words else 0 for word in vocabulary]

def vector_to_words(vector, vocabulary):
    return [vocabulary[i] for i, bit in enumerate(vector) if bit == 1]

def create_vocabulary(texts):
    vocab_set = set()
    for text in texts:
        vocab_set |= text_to_words(text)
    return sorted(vocab_set)


# ---------- CSV-Reader ----------
def read_csv_as_rec_of_lists(filename):
    rec = {}
    with open(filename, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key, value in row.items():
                rec.setdefault(key, []).append(value.strip())
    return rec


# ---------- Trainingsdaten ----------
class TrainingData:
    def __init__(self, input_list, output_list):
        self.input_list = input_list
        self.output_list = output_list
        self.input_vocab = create_vocabulary(input_list)
        self.output_vocab = create_vocabulary(output_list)

        self.input_vectors = [text_to_vector(txt, self.input_vocab) for txt in input_list]
        self.output_vectors = [text_to_vector(txt, self.output_vocab) for txt in output_list]

    def trainings_data_generator(self):
        for i, o, vi, vo in zip(self.input_list, self.output_list, self.input_vectors, self.output_vectors):
            yield {
                'input': i,
                'output': o,
                'v_input': vi,
                'v_output': vo,
            }


# ---------- Neuronales Netz ----------
class NNetwork:
    """einfaches implementiertes neuronales Netz mit sklearn"""
    def __init__(self, file_name='mlp_model.pkl'):
        self.file_name = file_name
        self.model = None
        self.vocabulary = None
        self.output_categories = None # Add this to store the unique output categories

    def load(self):
        with open(self.file_name, 'rb') as file:
            bundle = pickle.load(file)
        self.model = bundle['model']
        self.vocabulary = bundle['vocabulary']
        self.output_categories = bundle.get('outputs') # Load outputs, handling older files without it

    def save(self):
        with open(self.file_name, 'wb') as file:
            pickle.dump({
                'model': self.model,
                'vocabulary': self.vocabulary,
                'outputs': self.output_categories # Use the stored attribute here
                }, file)

    def teach(self, input_vectors, output_list, vocabulary):
        self.vocabulary = vocabulary
        # Store the unique output categories
        self.output_categories = list(set(output_list))
        hidden_size = len(input_vectors[0])
        self.model = MLPClassifier(hidden_layer_sizes=(hidden_size,), activation='relu', max_iter=1000)
        self.model.fit(np.array(input_vectors), np.array(output_list))
        self.save()

    def predict(self, input_vector):
        return self.model.predict(np.array(input_vector))
        
CSV_FILE_NAME = './ChatbotTraining.csv'
# ---------- Chatbot Funktionen ----------
def train_model():
    csv_data = read_csv_as_rec_of_lists(CSV_FILE_NAME)
    td = TrainingData(csv_data['patterns'], csv_data['tag'])

    nn = NNetwork()
    nn.teach(td.input_vectors, td.output_list, td.input_vocab)
    print("Training abgeschlossen. Modell gespeichert unter:", nn.file_name)

def run_prediction():
    nn = NNetwork()
    nn.load()
    print("Hallo. ich bin ein Chatbot. du kannst mir ein Satz sagen und ich werde versuchen zu predicten zum welches typ den Satz gehört.")
    print("Ich unterstütze die folgenden Typen:")
    print(nn.model.classes_)
    while True:
        user_input = input("Du:(exit zum Beenden) ").strip().lower()
        if user_input == 'exit':
            break
        input_vector = [text_to_vector(user_input, nn.vocabulary)]
        prediction = nn.predict(input_vector)
        print("Chatbot:", prediction[0])
        
        
def run_chatbot():
    """gleich wie run_prediction, aber mit zufälligen Antworten"""
    csv_data = read_csv_as_rec_of_lists(CSV_FILE_NAME)
    tags = csv_data['tag']
    answers = csv_data['responses']
    nn = NNetwork()
    nn.load()
    print("Hallo. ich bin ein Chatbot. Sag mir etwas")
    print("Ich unterstütze die folgenden Typen:")
    print(nn.model.classes_)
    while True:
        user_input = input("Du: ").strip().lower()
        if user_input == '':
            break
        input_vector = [text_to_vector(user_input, nn.vocabulary)]
        prediction = nn.predict(input_vector)
        print("Chatbot:", prediction[0])
        print("Antwort:", answers[tags.index(prediction[0])])

# ---------- Hauptprogramm ----------
if __name__ == '__main__':
    choice = input("Was möchtest du machen? \n(T)rainieren oder \n(P)Run in Prediction Mode\n(A)Run in Antwort Mode \n ('T', 'P' , 'A') : ").strip().lower()
    if choice == 't':
        train_model()
    elif choice == 'p':
        run_prediction()
    elif choice == 'a':
        run_chatbot()        
    else:
        print("Ungültige Eingabe. Bitte 'T' oder 'R' eingeben.")
