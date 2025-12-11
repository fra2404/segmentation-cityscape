# Cityscapes Semantic Segmentation

Implementazione di semantic segmentation su Cityscapes usando DeepLabV3 con backbone ResNet101.

## Struttura del Progetto

```
.
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── transforms.py      # Trasformazioni e mappings Cityscapes
│   │   └── dataset.py         # DataLoader e dataset handling
│   ├── models/
│   │   ├── __init__.py
│   │   └── deeplabv3.py       # Creazione e gestione modello
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py         # Logica di training
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py         # Metriche di valutazione
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Configurazione
│       └── visualization.py   # Visualizzazione risultati
├── train.py                   # Script principale per training
├── evaluate.py                # Script per evaluation
├── test_setup.py              # Test di verifica setup
├── requirements.txt           # Dipendenze Python
└── README.md                  # Questo file
```

## Installazione

1. **Clona il repository:**

```bash
cd artificial_intelligence_final_project
```

2. **Installa le dipendenze:**

```bash
pip install -r requirements.txt
```

3. **Verifica l'installazione:**

```bash
python test_setup.py
```

## Dataset

Scarica il dataset Cityscapes da [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/) e posizionalo in:

```
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

## Utilizzo

### Training

**Training base:**

```bash
python train.py
```

**Training con parametri personalizzati:**

```bash
python train.py \
    --num-epochs 10 \
    --batch-size 2 \
    --learning-rate 1e-3 \
    --image-size 1024 2048 \
    --device mps
```

**Training senza weighted sampler:**

```bash
python train.py --no-weighted-sampler
```

**Training solo su Frankfurt (validation):**

```bash
python train.py --filter-city frankfurt
```

### Evaluation

**Valutazione di un checkpoint:**

```bash
python evaluate.py --checkpoint ./checkpoints/best_model.pth
```

**Valutazione con visualizzazione:**

```bash
python evaluate.py \
    --checkpoint ./checkpoints/best_model.pth \
    --visualize \
    --num-samples 5
```

### Visualizzazione

**Solo visualizzazione (senza training):**

```bash
python train.py \
    --mode visualize \
    --load-checkpoint ./checkpoints/best_model.pth
```

## Parametri Principali

### Dati

- `--data-root`: Path al dataset Cityscapes (default: `./data/cityscapes`)
- `--batch-size`: Batch size (default: `2`)
- `--image-size`: Dimensione immagini come H W (default: `512 1024`)
- `--filter-city`: Filtra validation su città specifica (es. `frankfurt`)

### Training

- `--num-epochs`: Numero di epoche (default: `3`)
- `--learning-rate`: Learning rate (default: `1e-3`)
- `--weight-decay`: Weight decay (default: `1e-5`)
- `--grad-accum-steps`: Gradient accumulation steps (default: `2`)

### Sampling

- `--use-weighted-sampler`: Usa weighted sampling (default: attivo)
- `--no-weighted-sampler`: Disabilita weighted sampling
- `--max-samples-stats`: Max samples per statistiche (default: tutti)

### Device

- `--device`: Device da usare: `mps`, `cuda`, o `cpu` (default: `mps`)

### Checkpoint

- `--checkpoint-dir`: Directory per salvare checkpoint (default: `./checkpoints`)
- `--load-checkpoint`: Carica checkpoint esistente

### Mode

- `--mode`: Modalità: `train`, `eval`, o `visualize` (default: `train`)

## Architettura

- **Model**: DeepLabV3 con backbone ResNet101
- **Classes**: 19 classi Cityscapes
- **Loss**: CrossEntropyLoss con ignore_index=255
- **Optimizer**: AdamW
- **Scheduler**: StepLR (step_size=2, gamma=0.1)

## Features

✅ **Data Augmentation**: Flip, ColorJitter, Affine, Perspective  
✅ **Weighted Sampling**: Oversample classi rare  
✅ **Gradient Accumulation**: Supporto per batch size effettive maggiori  
✅ **BatchNorm Stability**: Fix per batch piccoli con MPS  
✅ **City Filtering**: Valutazione su città specifiche  
✅ **Checkpoint Management**: Salvataggio automatico best model  
✅ **Visualization**: Plot predizioni e training history  
✅ **Metrics**: Pixel accuracy e mIoU per classe

## Output

Durante il training vengono salvati:

- `checkpoints/checkpoint_epoch_N.pth`: Checkpoint per ogni epoca
- `checkpoints/best_model.pth`: Miglior modello (mIoU massimo)
- `training_history.png`: Grafici di loss e metriche
- `predictions.png`: Visualizzazioni (se richieste)

## Esempi di Risultati

Dopo il training, il modello produce:

- **Pixel Accuracy**: ~52-60% (dipende da epoche e parametri)
- **Mean IoU**: ~7-15% (migliora con più epoche e dati)

Le classi più performanti sono tipicamente: road, building, sky, vegetation, car.

## Note Tecniche

- **MPS Support**: Ottimizzato per Apple Silicon (M1/M2/M3)
- **Memory Efficient**: Batch size piccoli + gradient accumulation
- **Cityscapes Mappings**: Gestione corretta labelId → trainId
- **Ignore Index**: Corretto handling di pixel ignorate (255)

## Troubleshooting

**Errore CUDA/MPS non disponibile:**

```bash
python train.py --device cpu
```

**Out of memory:**

```bash
python train.py --batch-size 1 --grad-accum-steps 4
```

**Dataset non trovato:**
Verifica che `data/cityscapes/` contenga le cartelle corrette.

## Licenza

Questo progetto è per scopi educativi. Il dataset Cityscapes ha la sua licenza separata.
