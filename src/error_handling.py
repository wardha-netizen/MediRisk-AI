def get_prediction_error_message(error):
    """Return a user-friendly message for prediction errors."""

    if isinstance(error, FileNotFoundError):
        return (
            "The prediction model is currently unavailable. "
            "Please try again later."
        )

    if isinstance(error, (ValueError, TypeError, KeyError)):
        return (
            "The submitted information could not be processed. "
            "Please review the form and try again."
        )

    return (
        "We could not complete your prediction at this time. "
        "Please try again later."
    )