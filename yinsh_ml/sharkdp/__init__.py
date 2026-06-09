"""Bridge to the sharkdp/yinsh negamax engine.

The Rust engine lives in ``third_party/sharkdp_yinsh`` (cloned from
https://github.com/sharkdp/yinsh). We drive it through a small protocol
binary, ``yinsh-driver``, that speaks the *same* line protocol as
``third_party/yngine_driver`` — so this bridge reuses the yngine wire codec
(`yinsh_ml.yngine.move_codec`) verbatim and only swaps the binary path and the
search command (sharkdp searches to a fixed *depth*, not an MCTS sim budget).

Build the driver once::

    cd third_party/sharkdp_yinsh && cargo build --release -p yinsh_driver

Then::

    from yinsh_ml.sharkdp import Sharkdp
    with Sharkdp.start(depth=6) as eng:
        eng.apply(our_move)
        mv, wire = eng.get_move(player=state.current_player)
"""

from .bridge import Sharkdp, default_sharkdp_binary_path

__all__ = ["Sharkdp", "default_sharkdp_binary_path"]
