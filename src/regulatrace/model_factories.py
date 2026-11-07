"""Load optional local models only when the local inference profile requests them."""


def cross_encoder(*args, **kwargs):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(*args, **kwargs)


def sentence_transformer(*args, **kwargs):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(*args, **kwargs)
