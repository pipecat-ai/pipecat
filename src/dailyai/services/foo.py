def hello(p):
    def decorator(f):
        print("running f", p)
        return f

    return decorator

@hello("world")
def foo():
    print("hi")
