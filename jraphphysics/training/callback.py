class LogPyVistaPredictionsCallback:
    """
    Compatibility callback placeholder for JAX workflows.
    """

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def on_validation_epoch_end(self, *args, **kwargs):
        del args, kwargs
        return None
