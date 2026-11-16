"""`stampede-mcp`: the arena MCP server over stdio, for running on your own machine.

Finds the headline bank and the response tables either in a source checkout or where pip installs them
(<prefix>/share/stampede), then starts the same server as `python -m stampede.mcp_server.server`.
"""
import os
import site
import sys

from stampede.mcp_server import arena


def _data_root():
    checkout = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for root in (checkout, os.path.join(sys.prefix, "share", "stampede"), os.path.join(site.getuserbase(), "share", "stampede")):
        if os.path.isfile(os.path.join(root, "data", "headlines.json")):
            return root
    return None


def main():
    root = _data_root()
    if root is None:
        sys.exit("stampede-mcp: the headline bank and response tables were not found; reinstall the package")
    arena.HEADLINES = os.path.join(root, "data", "headlines.json")
    arena.SURR_DIR = os.path.join(root, "results", "surrogates")
    from stampede.mcp_server.server import main as serve

    serve()


if __name__ == "__main__":
    main()
