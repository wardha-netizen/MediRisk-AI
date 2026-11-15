import pytest

from src.error_handling import get_prediction_error_message


@pytest.mark.parametrize(
    "error, expected_message",
    [
        (
            FileNotFoundError(),
            "The prediction model is currently unavailable. Please try again later."
        ),
        (
            ValueError(),
            "The submitted information could not be processed. Please review the form and try again."
        ),
        (
            TypeError(),
            "The submitted information could not be processed. Please review the form and try again."
        ),
        (
            KeyError(),
            "The submitted information could not be processed. Please review the form and try again."
        ),
        (
            RuntimeError(),
            "We could not complete your prediction at this time. Please try again later."
        ),
    ]
)
def test_prediction_error_messages(error, expected_message):
    assert get_prediction_error_message(error) == expected_message