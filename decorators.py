import time

def time_function(string):
    def timeit(func):
        def wrapper(*args, **kwargs):
            a = time.time()
            res = func(*args)
            b = time.time()
            print(string, ' -> run time : ',str(b-a), 'sec')
        return wrapper
    return timeit

def print_function(name):
    def printer(func):
        def wrapper(*args, **kwargs):
            print('Executing function : {}'.format(str(name)) )
            res = func(*args)
        return wrapper
    return printer