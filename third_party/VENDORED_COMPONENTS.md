# Vendored components

| Component | Source | Recorded upstream commit | License |
|---|---|---|---|
| mcore-bridge | https://github.com/modelscope/mcore-bridge | `3591e434b7d5c3a4f1ae755794a5f83cd1b487cf` (exact import-tree match) | Apache-2.0 |
| vLLM | https://github.com/vllm-project/vllm | `20cac26b197a2e9530a787a5c1dbf5163533b2ff` | Apache-2.0 |
| lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness | `6632cb788facbbbe9b47a5133bfec6a67aee5ee8` (reconciled baseline) | MIT |
| FlashQLA Hopper subset | https://github.com/QwenLM/FlashQLA | `821fd9d37ede18fdc2a4e707fefe3770bfc32e58` | MIT |
| Optional research FLA 0.5.0 | https://github.com/fla-org/flash-linear-attention | `3a9ce1c83a13994d824dbb3421e2989d330bb38b` (source/wheel reconstruction verified) | MIT |
| Optional historical HSD FlashQLA subset | https://github.com/QwenLM/FlashQLA | `821fd9d37ede18fdc2a4e707fefe3770bfc32e58` | MIT |

Each directory retains LICENSE and a downstream NOTICE. UPSTREAM.md records
origin and local changes; the notices do not claim to be upstream-authored.
`dependency-manifest.json` pins every vendored file with SHA-256, including
downstream changes. It is a source inventory, not a complete transitive SBOM.

```bash
python scripts/march/check_vendor_manifest.py
```

mcore-bridge's original imported tree exactly matches its recovered upstream
revision. lm-evaluation-harness uses a reconciled baseline: 15,867 imported files
match, 9 differ and 33 are omitted, with the delta in its PROVENANCE.json. The
original evaluator clone SHA remains unknown and is not claimed as recovered.
Local import SHAs name private development history, not upstream commits.
File hashes establish integrity; license and transitive dependency review are
separate publication requirements.

[Maintenance boundaries](MAINTENANCE.md) distinguish the Safin-1 integration
from retained upstream catalogs and describe current ownership.

## Optional historical FLA research runtime

`flash-linear-attention/` reconstructs the FLA 0.5.0 library from pinned local wheel artifacts and restores the historical training modifications. It is selected only by research/hsd_gdn/run.py. LICENSE, NOTICE, UPSTREAM.md and PROVENANCE.json record attribution, wheel checksums and overlay fingerprints. Its upstream revision is verified against the official wheel contents and upstream namespace generator; see the [source provenance](flash-linear-attention/PROVENANCE.json) and [upstream record](flash-linear-attention/UPSTREAM.md). The recorded revision is not inferred from the version alone.
