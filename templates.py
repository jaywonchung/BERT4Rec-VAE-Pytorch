def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('train_bert'):
        args.mode = 'train'

        args.dataset_code = 'ml-1m'
        args.min_rating = 0
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'bert'
        batch = 128
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'bert'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 100
        args.metric_ks = [1, 5, 10, 20, 50]
        args.best_metric = 'NDCG@10'

        args.model_code = 'bert'
        args.model_init_seed = 0
        num_users, num_items = get_user_item_nums(args)

        args.bert_dropout = 0.1
        args.bert_hidden_units = 256
        args.bert_mask_prob = 0.15
        args.bert_max_len = 100
        args.bert_num_blocks = 2
        args.bert_num_heads = 4
        args.bert_num_items = num_items


def get_user_item_nums(args):
    if args.dataset_code == 'ml-1m':
        if args.min_rating == 4 and args.min_uc == 5 and args.min_sc == 0:
            return 6034, 3533
        elif args.min_rating == 0 and args.min_uc == 5 and args.min_sc == 0:
            return 6040, 3706
    raise ValueError()
