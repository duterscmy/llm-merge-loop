from .sleb import main_func as sleb_func
from .shortgpt import main_func as shortgpt_func
from .reverse import main_func as reverse_func
from .concat_merge import main_func as concat_merge_func
from .concat_merge_P import main_func as concat_merge_P_func
from .taylor import main_func as taylor_func
from .magnitude import main_func as magnitude_func


def mka_func(*args):
    from .mka import main_func

    return main_func(*args)

def CodeERROR(*args):
    raise NotImplementedError("Method not implemented")

methods_call = {"sleb": sleb_func,
                "shortgpt": shortgpt_func,
                "reverse": reverse_func,
                "mka": mka_func,
                "taylor": taylor_func,
                "taylor+": taylor_func,
                "magnitude": magnitude_func,
                "magnitude+": magnitude_func,
                "concat_merge": concat_merge_func,
                "concat_merge_P": concat_merge_P_func,
                "laco": CodeERROR,
                }
