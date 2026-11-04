# Test fixtures

`<game>_reply.md` — the model's released reply for Experiment 5 (design prose plus the `edits`
block that sets the weights). `<game>_hand.npz` — the network the authors read from that reply.
Both from https://github.com/RikiyaT/lm_understand-logs (CC BY 4.0; see ../../NOTICE.md).

They are the oracle for the edits-language parser: parsing `<game>_reply.md` must reproduce
`<game>_hand.npz` exactly.
