from pymoo.problems.nfv.graph import *

def get_problem(name, network=None, requests=None, *args, **kwargs):
    name = name.lower()

    if name.startswith("bbob-"):
        from pymoo.vendor.vendor_coco import COCOProblem
        return COCOProblem(name.lower(), **kwargs)
    
    from pymoo.problems.nfv.nfv import NFV
    PROBLEM = {
        'nfv': NFV
    }
    if name == "nfv":
        return NFV(network,requests)
    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name](*args, **kwargs)
