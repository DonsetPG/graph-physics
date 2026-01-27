from lightning.pytorch.callbacks import TQDMProgressBar


class ColabProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.disable = True
        return bar
