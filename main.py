from semantic_descriptor import SemanticDescriptor
from utils import closest_word

if __name__ == "__main__":
    semantic_descriptor = SemanticDescriptor()
    semantic_descriptor.train('data/train')
    semantic_descriptor.save('wap-sil-model-limited-punctuation.pkl')
    # from pprint import pprint
    
    # semantic_descriptor = SemanticDescriptor.load('wap-sil-model-limited-punctuation.pkl')
    accuracy = semantic_descriptor.evaluate('data/test/test.txt')
    print("accuracy:", accuracy)
    # print(semantic_descriptor.predict('downfall', 'ruin'))
    # print(semantic_descriptor.predict('downfall', 'hospital'))
    # print(semantic_descriptor.predict('downfall', 'font'))
    # print(semantic_descriptor.predict('downfall', 'gangster'))
    # downfall_synonym = closest_word(
    #     semantic_descriptor, 
    #     "downfall", 
    #     ["ruin", "hospital", "font", "gangster"])

    # option_synonym = closest_word(
    #     semantic_descriptor,
    #     "option",
    #     ["pity", "goof", "choice", "sickness"]
    # )

    # option_synonym = closest_word(
    #     semantic_descriptor,
    #     "vexed",
    #     ["pity", "goof", "choice", "sickness"]
    # )

    # print(option_synonym)

    



