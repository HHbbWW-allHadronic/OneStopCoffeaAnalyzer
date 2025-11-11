class AnalysisConfigurationError(Exception):
    def __init__(self, message):
        super().__init__(message)

class ResultIntegrityError(Exception):
    def __init__(self, message):
        super().__init__(message)

class AnalysisRuntimeError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)

