from transformers import Trainer, TrainingArguments, RobertaForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast, RobertaConfig

from mkhf.tokenizer import MKTokenizer


class MKTrainer(object):
    """ This class will generate a huggingface model for given data.

    Parameters
    ----------
    config: RobertaConfig
        This is the model configuration used by our model. Be sure to check vocab_size and set it as the number
        of unique courses in the dataset.

    Examples
    --------
    >>> from mkhf.trainer import MKTrainer
    >>> from mkhf.data_loader convert_table_to_str_files
    >>> convert_table_to_str_files('data.csv', './textdata')
    >>> trainer = MKTrainer()
    >>> trainer.train('./textdata', './Model')
    """

    def __init__(self, config: RobertaConfig = RobertaConfig(
            vocab_size=623,
            hidden_size=128,
            intermediate_size=256,
            max_position_embeddings=1536,
            num_attention_heads=8,
            num_hidden_layers=4),
            mlm_prob=0.4):
        super().__init__()
        self.config = config
        self.mlm_prob = mlm_prob

    def train(self, str_file_path: str, model_path='./Model', learning_rate=0.01, epochs=50, batch_size=128, block_size=200):
        """ Learn a surprise based model and save it as a pickle in a given path.

        Parameters
        ----------
        str_file_path: str
            The path containing train.txt and all.txt, gained from data_loader.
        model_path: str
            The path you want the model to be saved in.
        learning_rate: float
            The learning rate used for training BERT
        epochs: int
            Number of epochs over the dataset
        batch_size: int
            Size of each batch. It is recommended to use powers of two, e.g. 128, 265...
        block_size: int
            Number of courses to be taken into consideration for each history item
        """
        tokenizer = self.__init_tokenizer__(str_file_path, model_path)
        model = RobertaForMaskedLM(config=self.config)
        dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=f"{str_file_path}/train.txt",
            block_size=block_size,
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=self.mlm_prob
        )
        training_args = TrainingArguments(
            output_dir=f"{model_path}/checkpoints",
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset
        )
        trainer.train()
        trainer.save_model(model_path)

    def __init_tokenizer__(self, str_file_path: str, model_path='./Model'):
        _ = MKTokenizer(str_file_path, model_path)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{model_path}/vocab.json")
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.sep_token = "</s>"
        tokenizer.cls_token = "<s>"
        tokenizer.unk_token = "<unk>"
        tokenizer.pad_token = "<pad>"
        tokenizer.mask_token = "<mask>"
        return tokenizer
