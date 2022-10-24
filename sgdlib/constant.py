import sys
from threading import RLock


single_lock = RLock()


def Singleton(cls):
    instance = {}

    def _singleton_wrapper(*args, **kargs):
        with single_lock:
            if cls not in instance:
                instance[cls] = cls(*args, **kargs)
        return instance[cls]

    return _singleton_wrapper

@Singleton
class Const:
    class ConstValueError(PermissionError):
        pass
    class ConstCaseError(PermissionError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstValueError("Cannot modify the value of {0}.".format(name))

        if not name.isupper():
            raise self.ConstCaseError("Constant name {0} must be upper.".format(name))

        self.__dict__[name] = value


sys.modules[__name__] = Const()