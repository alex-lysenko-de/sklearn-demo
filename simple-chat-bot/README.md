               Chatbot
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
            2. Importiere den Datensatz genderclassification.csv in deine Python Datei
               und stelle es als pandas dataframe.
               Für den Rest braucht ihr die scikit-learn Library (sklearn)
               1. Datensatz in features und labels aufteilen
               2. Datensatz in Trainings- und Testdaten aufteilen
               3. DecisionTree Objekt erzeugen
               4. DecisionTree an die Trainingsdaten fitten
               5. Erstelle eine Vorhersage für die Testdaten 
               6. Gib die Accuracy aus

            3. Wende Schritt 1 und Schritt 2 an um den Datensatz ChatbotTraining.csv
               zu analysieren.
            4. Trainiere jetzt den ML Algorithmus mit dem gesamten Datensatz ChatbotTraining.csv
               und teste ihn, indem du einen Satz im Terminal eingibst,dieser in
               ein BoW verwandelt wird und anschließend dieser von der
               predict()-Methode klassifiziert wird und die Klasse ausgibt.
            5. Wie Schritt 4, aber anstatt dass die Kategorie am Ende ausgegeben wird, 
               wird die Information genutzt um eine Antwort aus der response-Spalte mit
               passender Kategorie zufällig auszuwählen.## Chatbot
