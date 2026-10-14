from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer


sw = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')


def process_text(review, stem='p'):
    """
    Given a text, the function converts the text into lower case,
    removes stopwords, removes punctuations, tokenizes the text,
    performs stemming and returns the processed text
    :param review: raw text
    :param stem: Stemmer - 'p' for PorterStemmer and 'l' for
                        LancasterStemmer
    :return: processed text
    """
    # Convert text to lower
    review = review.lower()
    # Word tokenize the review
    tokens = word_tokenize(review)
    # Remove stopwords
    tokens = [t for t in tokens if t not in sw]
    # Remove punctuation
    tokens = [tokenizer.tokenize(t) for t in tokens]
    tokens = [t for t in tokens if len(t)>0]
    tokens = ["".join(t) for t in tokens]
    # Create stemmer
    if stem == 'p':
        stemmer = PorterStemmer()
    elif stem == 'l':
        stemmer = LancasterStemmer()
    else:
        raise Exception("stem has to be either 'p' for Porter or 'l' for Lancaster")
    # Stemming
    tokens = [stemmer.stem(t) for t in tokens]
    # Return clean string
    return " ".join(tokens)
