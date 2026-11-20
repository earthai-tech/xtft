# data/

Datasets used by the benchmarks and the case study. Only this README
is tracked by git; the data themselves stay out of the repository.

| Folder | Content | Local storage | HPC storage |
|---|---|---|---|
| `raw/` | downloads as distributed (datasetsforecast long-horizon archive, ...) | `E:\_DATA\xtft\raw` (junction) | `$XTFT_SCRATCH/data` |
| `processed/` | derived panels (Nansha/Zhongshan Stage-1 exports, ...) | `E:\_DATA\xtft\processed` (junction) | `$XTFT_SCRATCH/processed` |

`benchmarks/datasets.py` reads and caches into `data/raw` unless the
`XTFT_DATA` environment variable points elsewhere (the SLURM scripts
set it to the scratch path).

Recreate the junctions on a new Windows machine (no admin rights
needed):

```powershell
New-Item -ItemType Junction -Path data\raw -Target E:\_DATA\xtft\raw
New-Item -ItemType Junction -Path data\processed -Target E:\_DATA\xtft\processed
```

On Linux or macOS use symbolic links (`ln -s`) or plain folders.
