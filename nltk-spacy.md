# spaCy and NLTK

Tokenization, POS tagging, dependency parse, lemmatization, sentence boundary detection.

## NLTK

Tokenization:
 - PunktSentenceTokenizer: Separate document into sentences
 - TreebankWordTokenizaiton
 - WordPunctTokenizer
 - PunctWordTokenizer
 - WhitespaceTokenizer

Text Cleaning:
 - `isalpha` - does a string contain alphabetical characters? If not, discard
    the numbers and punctuation etc.
 - FreqDist: See the frequency of words. Keep only the words that appear more than N times.
 - StopWords: Most frequent english stopwords, we can just remove them.
   - But for some problems removing stopwords is not such a great idea. Maybe you can remove
     stopword POS tags.
   - If you don't know what to do, better not to remove them.

Bag of Words (converting words into numerical values):
 - Just assign each word a number, 

## spaCy

Need to download your own language model.

Tokenizer:
 - Iterate over space spearated subtrings
 - Check wehterh we have an explicitly defined rule 
 - Consume prefix, consume suffix, consume infix - then it becomes a single token
 - Can't use FreqDist, but Counter is similar but counts things slightly differently.

Word Embeddigns:
 - Iterate over tokens lookup the corresponding pretrained word2vec.
 - Some words don't have vectors

Speech Tagging:
 - Analyze the grammatical structure of a sentence, what is related to what and how
 - Noun chunks: Best noun phrases that have a noun as their head
 - Use NER to replace dates and place names if necessary. reduces dimensionality

