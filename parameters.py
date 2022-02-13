class ProjectParams():
    def __init__(self):
        
        # Subset of sequences you want to use
        self.training_sequences = {'00': ['000', '004540'],
                                   '01': ['000', '001100']}
                                   #'02': ['000', '004660'],
                                   #'05': ['000', '002760'],
                                   #'08': ['001100', '005170'],
                                   #'09': ['000', '001590']}
        
        self.validation_sequences = {'04': ['000', '000270']}
                                     #'06': ['000', '001100'],
                                     #'07': ['000', '001100'],
                                     #'10': ['000', '001200']}
        
        self.sequences = self.training_sequences
        self.sequences.update(self.validation_sequences)
        
        self.partition = None  # partition videos in 'train_video' to train / valid dataset  #0.8
        self.seq_len = (5, 10)
        self.sample_times = 4
        
        # Data info path
        self.train_data_info_path = 'datainfo/train_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(
            ''.join(self.training_sequences.keys()), ''.join(self.validation_sequences.keys()),
            self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)
        self.valid_data_info_path = 'datainfo/valid_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(
            ''.join(self.training_sequences.keys()), ''.join(self.validation_sequences.keys()),
            self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)
        
        # S3 Bucket Info
        self.bucket = 'deepvo-data'
        self.sequence_key = 'sequences/'
        self.pose_key = 'poses/'
        self.sequence_location = 's3://{}/{}'.format(self.bucket, self.sequence_key)
        self.pose_location = 's3://{}/{}'.format(self.bucket, self.pose_key)
        
        # Data Preprocessing
        self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
        self.img_w = 608   # original size is about 1226
        self.img_h = 184   # original size is about 370
        # Tensors from preprocessing
        self.minus_point_5 = True
        self.img_means = (-0.1462488382606457, -0.1239967719900155, -0.1208675856721287)
        self.img_stds = (0.2909732338687599, 0.3071966735493643, 0.3264705248159934)
        
        # DeepVO Model
        self.rnn_hidden_size = 1000
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0   # 0: no dropout
        self.clip = None
        self.batch_norm = True
        
        # Training
        self.n_processors = 4
        self.epochs = 50
        self.batch_size = 5
        self.pin_mem = True
        self.optim = {'opt': 'Adagrad', 'lr': 0.0005}
                    # Choice:
                    # {'opt': 'Adagrad', 'lr': 0.001}
                    # {'opt': 'Adam'}
                    # {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}
        
        # Flownet
        self.pretrained_flownet = "flownets_EPE1.951.pth.tar"
                                # Choice:
                                # None
        self.resume = False  # resume training
        self.resume_t_or_v = '.train'
        
        # Paths on Paths
        self.load_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(
            ''.join(self.training_sequences.keys()), ''.join(self.validation_sequences.keys()),
            self.img_h, self.img_w,
            self.seq_len[0], self.seq_len[1],
            self.batch_size,
            self.rnn_hidden_size,
            '_'.join([k+str(v) for k, v in self.optim.items()]),
            self.resume_t_or_v)
        
        self.load_optimizer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(
            ''.join(self.training_sequences.keys()), ''.join(self.validation_sequences.keys()),
            self.img_h, self.img_w,
            self.seq_len[0], self.seq_len[1],
            self.batch_size,
            self.rnn_hidden_size,
            '_'.join([k+str(v) for k, v in self.optim.items()]),
            self.resume_t_or_v)

        self.record_path = 'records/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(
            ''.join(self.training_sequences.keys()), ''.join(self.validation_sequences.keys()),
            self.img_h, self.img_w,
            self.seq_len[0], self.seq_len[1],
            self.batch_size,
            self.rnn_hidden_size,
            '_'.join([k+str(v) for k, v in self.optim.items()]))
        
        self.save_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(
            ''.join(self.training_sequences.keys()), ''.join(self.validation_sequences.keys()),
            self.img_h, self.img_w,
            self.seq_len[0], self.seq_len[1],
            self.batch_size,
            self.rnn_hidden_size,
            '_'.join([k+str(v) for k, v in self.optim.items()]))
        
        self.save_optimzer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(
            ''.join(self.training_sequences.keys()), ''.join(self.validation_sequences.keys()),
            self.img_h, self.img_w,
            self.seq_len[0], self.seq_len[1],
            self.batch_size,
            self.rnn_hidden_size,
            '_'.join([k+str(v) for k, v in self.optim.items()]))


par = ProjectParams()