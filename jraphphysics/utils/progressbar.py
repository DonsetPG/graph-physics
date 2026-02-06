class ColabProgressBar:
    """
    Compatibility stub for environments not using Lightning.
    """

    def init_validation_tqdm(self):
        return None

    def init_predict_tqdm(self):
        return None

    def init_test_tqdm(self):
        return None
