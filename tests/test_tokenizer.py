from src.indexer import tokenize

def test_tokenize_basic():
    text = "Hello world! Test 2026."
    tokens = list(tokenize(text))
    assert "hello" in tokens
    assert "world" in tokens
    assert "2026" in tokens

def test_tokenize_apostrophe_and_hyphen():
    text = "it's well-known test"
    tokens = list(tokenize(text))
    assert "it's" in tokens
    assert "well-known" in tokens