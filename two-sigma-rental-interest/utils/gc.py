import gc
import torch

def gc_and_clear_caches(value):
    """A simple pass-through funcion that clears caches."""
    gc.collect()
    torch.cuda.empty_cache()
    
    return value
