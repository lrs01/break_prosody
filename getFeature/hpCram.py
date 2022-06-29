


class hpCram():
    def  __init__(self):
        self.num_mels = 80
        # num_freq = 1024
        self.n_fft = 2048
        self.sr = 16000
        # frame_length_ms = 50.
        # frame_shift_ms = 12.5
        self.preemphasis = 0.97
        self.frame_shift = 0.0125 # seconds
        self.frame_length = 0.05 # seconds
        self.hop_length = int(self.sr*self.frame_shift) # samples.
        self.win_length = int(self.sr*self.frame_length) # samples.
        self.n_mels = 80 # Number of Meƒl banks to generate
        self.power = 1.2 # Exponent for amplifying the predicted magnitude
        self.min_level_db = -100
        self.ref_level_db = 20
        self.hidden_size = 256
        self.embedding_size = 512
        self.max_db = 100
        self.ref_db = 20
        self.preemphasis=0.97