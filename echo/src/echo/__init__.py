"""echo — the memory organ.

Machine listening as a persistent archive: every sound heard is encoded
into a dense, temporal feature vector and stored in a vector database
that answers the only question novelty search actually asks:

    HAVE WE HEARD SOMETHING LIKE THIS BEFORE?

novelty(p) = mean distance to the k nearest neighbors in the archive
(Lehman & Stanley, 2011). The archive is the world; the encoder decides
what "like" means. Everything is offline, pure numpy, deterministic.
"""

from .encode import (LAYOUT, DIM, BLOCK_NAMES, encode, encode_blocks,
                     N_CHUNK, N_MFCC)
from .bank import EchoBank, EchoMetric

__all__ = ["LAYOUT", "DIM", "BLOCK_NAMES", "encode", "encode_blocks",
           "N_CHUNK", "N_MFCC", "EchoBank", "EchoMetric"]

__version__ = "0.1.0"