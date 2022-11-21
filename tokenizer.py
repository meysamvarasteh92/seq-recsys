from tokenizers.pre_tokenizers import Whitespace, Digits, Sequence
from tokenizers.models import WordLevel
from tokenizers.processors import BertProcessing
from tokenizers.trainers import WordLevelTrainer

from tokenizers import Tokenizer


class MKTokenizer(object):
    def __init__(self,  str_file_path: str, model_path='./Model'):
        super().__init__()
        tokenizer = Tokenizer(WordLevel())

        pre_tokenizer = Sequence([Whitespace(), Digits()])
        tokenizer.pre_tokenizer = pre_tokenizer

        trainer = WordLevelTrainer(
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
        tokenizer.train(files=[f'{str_file_path}/all.txt'], trainer=trainer)

        tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )

        tokenizer.save(f"{model_path}/vocab.json")
