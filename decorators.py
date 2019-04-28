import time

def time_function(name):
    def timeit(func):
        def wrapper(*args, **kwargs):
            a = time.time()
            res = func(*args)
            b = time.time()
            print('{} -> {} sec'.format(name, str(b-a)))
            return res
        return wrapper
    return timeit

def print_function(name):
    def printer(func):
        def wrapper(*args, **kwargs):
            print('Executing function : {}'.format(str(name)) )
            res = func(*args)
            return res
        return wrapper
    return printer
