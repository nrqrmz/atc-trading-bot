import warnings


class PipelineWarning(UserWarning):
    """Warning issued when a pipeline step is called out of order."""
    pass


_original_formatwarning = warnings.formatwarning


def _format_pipeline_warning(message, category, filename, lineno, line=None):
    if issubclass(category, PipelineWarning):
        return f"Warning: {message}\n"
    return _original_formatwarning(message, category, filename, lineno, line)


warnings.formatwarning = _format_pipeline_warning
