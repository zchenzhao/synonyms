import os

from semantic_descriptor import WordNotFoundException

def load_text(directory):
    text = ""

    print(f"files to load: {os.listdir(directory)}")

    for filepath in os.listdir(directory):
        with open(os.path.join(directory, filepath), "r", encoding="latin1") as f:
            text = " ".join([text, f.read()])

    return text

def closest_word(model, word_to_match, potential_synonyms):
    highest_similarity = float('-inf')
    closest_synonym = None

    for potential_synonym in potential_synonyms:
        try:
            similarity = model.predict(word_to_match, potential_synonym)

            if similarity > highest_similarity:
                highest_similarity = similarity
                closest_synonym = potential_synonym
        except WordNotFoundException:
            pass

    return closest_synonym


        