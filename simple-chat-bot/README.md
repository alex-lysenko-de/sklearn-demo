# Chatbot

## 1. Bag of Words:

```python
vokabular = [python, ich, wie, geht, liebe, dir, es]
satz1 = ich liebe python
satz2 = hallo wie geht es dir denn
bow1 = [1, 1, 0, 0, 1, 0, 0]
bow2 = [0, 0, 1, 1, 0, 1, 1]
````

---

## 2. Machine Learning mit `genderclassification.csv`

Importiere den Datensatz `genderclassification.csv` in deine Python-Datei und stelle ihn als pandas DataFrame dar.

Für den Rest brauchst du die scikit-learn Library (`sklearn`):

1. Datensatz in Features und Labels aufteilen
2. Datensatz in Trainings- und Testdaten aufteilen
3. DecisionTree-Objekt erzeugen
4. DecisionTree an die Trainingsdaten fitten
5. Eine Vorhersage für die Testdaten erstellen
6. Die Accuracy ausgeben

---

## 3. Analyse des Datensatzes `ChatbotTraining.csv`

Wende **Schritt 1** (BoW) und **Schritt 2** (ML mit scikit-learn) auf den Datensatz `ChatbotTraining.csv` an.

---

## 4. Interaktiver Chatbot

Trainiere jetzt den ML-Algorithmus mit dem gesamten Datensatz `ChatbotTraining.csv`
und teste ihn, indem du einen Satz im Terminal eingibst.
Dieser Satz wird in ein BoW umgewandelt und anschließend von der `predict()`-Methode klassifiziert.
Gib die zugehörige Klasse aus.

---

## 5. Chatbot mit Antwort

Wie Schritt 4, aber anstatt dass die Kategorie am Ende ausgegeben wird,
wird die Information genutzt, um eine Antwort aus der `response`-Spalte mit
passender Kategorie **zufällig auszuwählen**.

