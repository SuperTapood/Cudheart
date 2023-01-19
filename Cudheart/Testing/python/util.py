import numpy as np

def check(res, output):
    close = np.allclose(res, output)
    
    if type(res) == np.ndarray:
        res = res.tolist()
    
    mark = "T" if close else "F"
    
    out = str(res) + mark
    
    print(out, end="")