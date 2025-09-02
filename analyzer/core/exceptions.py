class AnalysisConfigurationError(Exception):
    def __init__(self, message):
        super().__init__(message)

class ResultIntegrityError(Exception):
    def __init__(self, message):
        super().__init__(message)

class MultiTaskException(Exception):
    def __init__(self, message, exceptions):
        super().__init__(message)
        self.exceptions = exceptions
