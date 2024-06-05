import multiprocessing as mpr
import re
import string
from copy import copy
from typing import Iterable, Union, List

import pymorphy2
from gensim.models import FastText
from nltk.corpus import stopwords
from razdel import sentenize, tokenize
from tqdm import tqdm


N_CORES = mpr.cpu_count()

fasttext_lowercase_train_config = {
    "language": "russian",
    "remove_stopwords": True,
    "remove_numbers": True,
    "remove_punctuations": True,
    "remove_extra_spaces": True,
    "lowercase": True,
    "lemmatization": True,
    # model training config
    "model": FastText,
    "epochs": 5,
    "model_params": {
        "vector_size": 100,
        "window": 5,
        "min_count": 1,
        "workers": N_CORES - 1
    }
}

file_path = 'russian_stop_words.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    custom_stop_words = file.read().splitlines()


class TextPreprocessor:
    """
    Class for text preprocessing
    """

    # config key-names
    REMOVE_PUNCTUATIONS = "remove_punctuations"
    REMOVE_NUMBERS = "remove_numbers"
    REMOVE_STOPWORDS = "remove_stopwords"
    CUSTOM_STOPWORDS = "custom_stopwords"
    REMOVE_EXTRA_SPACES = "remove_extra_spaces"
    LOWERCASE = "lowercase"
    LEMMATIZATION = "lemmatization"
    YO_REPLACEMENT = "yo_replacement"
    ONLY_LETTERS = "only_letters"

    def __init__(self, config: dict):
        self.language = "russian"
        
        self.no_stopwords = config.get(self.REMOVE_STOPWORDS, False)
        self.custom_stopwords = config.get(self.CUSTOM_STOPWORDS, list())
        self.no_numbers = config.get(self.REMOVE_NUMBERS, True)
        self.no_punctuations = config.get(self.REMOVE_PUNCTUATIONS, True)
        self.no_extra_spaces = config.get(self.REMOVE_EXTRA_SPACES, True)
        self.lower_text = config.get(self.LOWERCASE, True)
        self.lemmatize = config.get(self.LEMMATIZATION, False)
        self.yo_replacement = config.get(self.YO_REPLACEMENT, True)
        self.only_letters = config.get(self.ONLY_LETTERS, False)

        self.nlp_model = pymorphy2.MorphAnalyzer()
        self.stopwords = set(stopwords.words('russian')) | set(self.custom_stopwords)

        if self.no_stopwords:
            self.stopwords = set(stopwords.words(self.language) + self.custom_stopwords)

    @staticmethod
    def remove_extra_spaces(text: Union[str, List[str]]) -> Union[str, List[str]]:
        def _remove_extra_spaces(string):
            return re.sub(r"\s{2,}", " ", string).strip()

        if isinstance(text, str):
            return _remove_extra_spaces(text)
        if isinstance(text, Iterable):
            return [_remove_extra_spaces(token) for token in text]
        raise TypeError(f"Type {type(text)} is not supported.")

    @staticmethod
    def remove_empty_tokens(text: Union[str, List[str]]):
        if isinstance(text, str):
            return text

        return [
            token
            for token in text
            if re.sub(r"\s+", " ", token, re.IGNORECASE).strip() != ""
        ]

    @staticmethod
    def remove_numbers(text: Union[str, List[str]]) -> Union[str, List[str]]:
        def _remove_numbers(string):
            return re.sub(r"\d+\w*", "", string)

        if isinstance(text, str):
            return _remove_numbers(text)
        if isinstance(text, Iterable):
            return [_remove_numbers(token) for token in text]
        raise TypeError(f"Type {type(text)} is not supported.")

    @staticmethod
    def remove_punctuations(text: Union[str, List[str]]) -> Union[str, List[str]]:
        def _remove_punctuations(str_text):
            return re.sub(
                "[%s]" % re.escape(string.punctuation + "№«»’·⋅…"), " ", str_text
            )

        if isinstance(text, str):
            return _remove_punctuations(text)
        if isinstance(text, Iterable):
            return [_remove_punctuations(token) for token in text]
        raise TypeError(f"Type {type(text)} is not supported.")

    @staticmethod
    def make_text_lower(text: Union[str, Iterable[str]]) -> Union[str, Iterable[str]]:
        if isinstance(text, str):
            return text.lower()
        if isinstance(text, Iterable):
            return [token.lower() for token in text]
        raise TypeError(f"Type {type(text)} is not supported.")

    @staticmethod
    def replace_yo(text: Union[str, Iterable[str]]) -> Union[str, Iterable[str]]:
        if isinstance(text, str):
            return text.replace("ё", "е").replace("Ё", "Е")
        if isinstance(text, Iterable):
            return [token.replace("ё", "е").replace("Ё", "Е") for token in text]
        raise TypeError(f"Type {type(text)} is not supported.")

    def remove_stopwords(
        self, text: Union[str, Iterable[str]]
    ) -> Union[str, Iterable[str]]:
        raw_text = copy(text)
        if isinstance(text, str):
            raw_text = text.split()

        try:
            text_without_stopwords = []
            for word in raw_text:
                word = word.lower()
                if word not in self.stopwords and len(word) > 2:
                    text_without_stopwords.append(word)

            return (
                text_without_stopwords
                if not isinstance(text, str)
                else " ".join(text_without_stopwords)
            )

        except AttributeError:
            print(
                "To remove stopwords - configuration of 'remove_stopwords' is required!"
            )
            return text

    @staticmethod
    def remove_all_except_letters(text: Union[str, Iterable[str]]):
        if isinstance(text, str):
            return re.sub("[^A-Za-zА-Яа-я]+", " ", text)
        if isinstance(text, Iterable):
            return [re.sub("[^A-Za-zА-Яа-я]+", " ", token) for token in text]
        raise TypeError(f"Type {type(text)} is not supported.")

    def text_cleaning(
        self, text: Union[str, Iterable[str]]
    ) -> Union[str, Iterable[str]]:
        assert isinstance(text, str) or isinstance(
            text, Iterable
        ), f"{'Input text must be string or iterable, not'} {type(text)}"

        text = self.make_text_lower(text) if self.lower_text else text
        text = self.replace_yo(text) if self.yo_replacement else text
        text = self.remove_numbers(text) if self.no_numbers else text
        text = self.remove_punctuations(text) if self.no_punctuations else text
        text = self.remove_all_except_letters(text) if self.only_letters else text
        text = self.remove_extra_spaces(text) if self.no_extra_spaces else text
        text = self.remove_empty_tokens(text)
        text = self.remove_stopwords(text) # if self.no_stopwords else text
        text = self.text_lemmatization(text) if self.lemmatize else text

        return text

    def _russian_lemmatization(self, text: Union[str, Iterable[str]]) -> Iterable[str]:
        def lemmatize(word):
            return self.nlp_model.parse(word)[0].normal_form

        if isinstance(text, str):
            text = text.split()

        return [lemmatize(word) for word in text]


    def text_lemmatization(
        self, text: Union[str, Iterable[str]]
    ) -> Union[str, Iterable[str]]:
        lemmas = self._russian_lemmatization(text=text)

        return " ".join(lemmas) if isinstance(text, str) else lemmas
