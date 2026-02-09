MAX_TEXT_LENGTH = 10000  # Prevent abuse or overload

def validate_input(text: str) -> str:
    """
    Ensures safe and valid input
    """

    if not text or not isinstance(text, str):
        raise ValueError("Invalid input text")

    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError("Input text too long")

    # Strip excessive whitespace
    text = text.strip()

    return text
