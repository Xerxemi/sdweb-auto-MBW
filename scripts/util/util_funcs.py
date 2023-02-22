def grouped(iterable, n):
        return zip(*[iter(iterable)]*n)
