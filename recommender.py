from typing import List

from transformers import pipeline
from transformers import PreTrainedTokenizerFast


class MKRecommender(object):
    """ This class will use an already trained model and will generate items for users.

    Parameters
    ----------
    model_path : str
        The path to generated model by MKTrainer.
    data_path : str
        The path to your data. This will be used to get a history of each user's interactions.
    filter_zeros: bool
        Whether 

    Examples
    --------
    >>> from mkhf.recommender import MKRecommender
    >>> recommender = MKRecommender('./Model')
    >>> recoms = recommender.recommend('263 347 401')
    >>> for recom in recoms:
    >>>     print(recom)
    """

    def __init__(self, model_path: str):
        super().__init__()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f'{model_path}/vocab.json')
        tokenizer.bos_token = '<s>'
        tokenizer.eos_token = '</s>'
        tokenizer.sep_token = '</s>'
        tokenizer.cls_token = '<s>'
        tokenizer.unk_token = '<unk>'
        tokenizer.pad_token = '<pad>'
        tokenizer.mask_token = '<mask>'

        self.fill_mask = pipeline(
            'fill-mask',
            model=model_path,
            tokenizer=tokenizer
        )

    def recommend(self, history: str, size=10) -> List[str]:
        """ Will recommend a list of courses for the given history.

        Parameters
        ----------
        history : str
            The history of courses a user has visited, separated by spaces. for example: 263 347 401
        size : int
            The number of items to be recommended

        Returns
        -------
        response : List
            A list of required length, containing recommended courses.
        """
        query = f'{history} <mask>'
        response = [x['token_str'] for x in self.fill_mask(query, top_k=size)]
        return response
