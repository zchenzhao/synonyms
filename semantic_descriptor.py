import os

class SemanticDescriptor(object):

    def __init__(self):
        self.mapping = {}

    @staticmethod
    def load(filepath):
        import pickle 

        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def save(self, filename, directory=os.getcwd()):
        import pickle

        with open(os.path.join(directory, filename), 'wb') as f:
            pickle.dump(self, f)

    def train(self, training_directory):
        from utils import load_text
        
        text = load_text(training_directory)
        sentences = self._preprocess(text)

        # what if the same word appears twice in one sentence? do we discount that?
        for sentence in sentences:
            for word in sentence:
                self._update_mapping(word, sentence)

    def predict(self, word_1, word_2):
        import pandas as pd
        from sklearn.metrics.pairwise import cosine_similarity

        word_1 = word_1.lower()
        word_2 = word_2.lower()

        vector_1 = self.mapping.get(word_1)
        vector_2 = self.mapping.get(word_2)

        if vector_1 is None:
            raise WordNotFoundException(f"`{word_1}` not found.")
        if vector_2 is None:
            raise WordNotFoundException(f"`{word_2}` not found.")

        for word in vector_1:
            if word not in vector_2:
                vector_2[word] = 0

        for word in vector_2:
            if word not in vector_1:
                vector_1[word] = 0

        vector_1 = pd.Series(vector_1).sort_index()
        vector_2 = pd.Series(vector_2).sort_index()

        return cosine_similarity([vector_1], [vector_2])[0][0]

    def evaluate(self, test_file):
        from utils import closest_word

        with open(test_file, 'r') as f:
            questions = f.readlines()

        number_of_questions = len(questions)
        number_correct = 0
        
        for question in questions:
            question = question.strip().split()

            word_to_match = question[0]
            correct_answer = question[1]
            prediction = closest_word(self, word_to_match, question[2:])

            if correct_answer == prediction:
                number_correct += 1
            # else:
                # print(question)
                # print(f"word_to_match:", word_to_match)
                # print(f"correct_answer:", correct_answer)
                # print(f"prediction:", prediction)

        accuracy = number_correct / number_of_questions

        print("number_of_questions:",number_of_questions)

        return accuracy

    def _update_mapping(self, word, sentence):
        for context_word in sentence:
            if word != context_word:
                if word not in self.mapping:
                    self.mapping[word] = {}

                if context_word in self.mapping[word]:
                    self.mapping[word][context_word] += 1
                else:
                    self.mapping[word][context_word] = 1

    def _preprocess(self, text):
        import re
        
        text = text.lower()
        punctuation_stripped_text = re.sub("[-;:,]", "", text)
        sentences = re.split("\?+|\!+|\.+", punctuation_stripped_text)

        sentences = [sentence.split() for sentence in sentences]

        return sentences

class WordNotFoundException(Exception):
    pass


