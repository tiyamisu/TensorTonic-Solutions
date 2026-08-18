def log_transform(values):
    return [__import__('math').log1p(x) for x in values]