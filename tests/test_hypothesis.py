from hypothesis import given, strategies as st


@given(st.lists(st.integers(min_value=1, max_value=1000), unique=True))
def test_postings_sorting_property(doc_ids: list[int]) -> None:
    """Властивість: перевірка сортування ідентифікаторів документів."""
    sorted_ids = sorted(doc_ids)
    assert sorted_ids == sorted(sorted_ids)


@given(st.text(min_size=1, max_size=50))
def test_string_non_empty_identity(text: str) -> None:
    """Властивість: базові перетворення рядків для генератора тестів."""
    assert str(text) == text


@given(st.integers(min_value=-100, max_value=100), st.integers(min_value=-100, max_value=100))
def test_addition_commutative(a: int, b: int) -> None:
    """Властивість: комутативність додавання для числових генераторів."""
    assert a + b == b + a