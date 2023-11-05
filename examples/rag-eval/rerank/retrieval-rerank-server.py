from typing import List
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

from FlagEmbedding import FlagReranker

# reranker = FlagReranker("BAAI/bge-reranker-large", use_fp16=True)
reranker = FlagReranker("BAAI/bge-reranker-base") # cpu can not set use_fp16

def reranker_compute_score(query: str, docs: list[str]) -> List[float]:
    return reranker.compute_score([(query, i) for i in docs])


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ("/RPC2",)

# Create server
with SimpleXMLRPCServer(("localhost", 8000),
                        requestHandler=RequestHandler) as server:
    server.register_introspection_functions()

    server.register_function(reranker_compute_score) # type: ignore

    # Run the server's main loop
    server.serve_forever()
