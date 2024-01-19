- ACOBS:
```
    python run_seed_sgaco.py --n_jobs 30 \
                            --output_dir ./experiment/ACOBS \
                         
```

- IEGWO:
```
    python run_seed_iegwo.py --n_jobs 30 \
                            --output_dir ./experiment/IEGWO \
                         
```

- EmcFIS:
```
    python run_seed_emcfis.py --n_jobs 30 \
                            --output_dir ./experiment/GA_EmcFIS \
    python run_seed_mcfis.py --n_jobs 30 \
                            --input_dir ./experiment/GA_EmcFIS \
                            --output_dir ./experiment/EmcFIS \
                         
```