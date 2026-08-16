def text_chunking(tokens, chunk_size, overlap):
    if not tokens:
        return []

    step = chunk_size - overlap

    if len(tokens) <= chunk_size:
        return [tokens]

    chunks = []

    for i in range(0, len(tokens) - chunk_size + 1, step):
        chunks.append(tokens[i:i + chunk_size])

    return chunks