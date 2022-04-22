from sacred import Experiment

ex = Experiment('Intent_CLS', save_git_info=False)

@ex.config
def config():
    exp_name = "Intent_CLS"
    model_name = 'roberta-base'
    
    seed = 1841
    batch_size = 64
    input_seq_len = 50
    max_steps = 5000
    warmup_steps = 500
    
    num_workers = 5
    gpus = [0,1,2,3]
    val_check_interval = 0.5
    load_path = ''
    log_dir = 'result'
    train_dataset_path = 'data/HWU64/train'
    val_dataset_path = 'data/HWU64/valid'
    test_dataset_path = 'data/HWU64/test'

# text encoder
@ex.named_config
def text_roberta_base():
    tokenizer = "roberta-base"
    vocab_size = 50265
    input_text_embed_size = 768
    