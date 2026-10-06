# Krisna â€” Agentic Design System

> [!IMPORTANT]
> **Canonical Master Document**: The formal, authoritative system specification is [docs/PRD.md](docs/PRD.md) (Product & Research Requirements Document â€” Final No-RLHF Revision). All architecture, hardware envelopes, state machines, and citations conform strictly to it.

A conversational, agentic UI-design system: a Planner discusses intent, a
Sketch tier generates a partial draft, a Polish tier finalizes a
high-fidelity render, and an on-demand Critic tier reviews it. This
monorepo covers the full pipeline from raw public datasets to a running
inference service, plus a working control-panel UI over that service.

**Final architecture (no-RLHF-loop PRD revision)**: the Planner
(Qwen3.5-9B) and Qwen-Image-Edit-2511 ship **frozen** â€” RAG retrieval and
zero-shot ICL respectively, no fine-tuning. Gemma-4 (Critic) ships
**frozen** as an on-demand product feature. Only the **Sketch tier**
(trained from scratch) and **Z-Image-Turbo** (LoRA + Diffusion-DPO) are
actually trained. There is no AI-judge-labeled training data anywhere in
this repository â€” every training signal is either a real public dataset
or a frozen model needing no training data at all.

## Layout

```
krisna/
â”œâ”€â”€ docs/                # all documentation â€” start at docs/README.md,
â”‚                           and docs/review/README.md for the full,
â”‚                           phase-by-phase design-sync audit
â”œâ”€â”€ scripts/              # all shell scripts, split by package
â”œâ”€â”€ tests/                # ALL tests, root-level â€” pytest tests/ runs everything
â”œâ”€â”€ models/                # trained checkpoint artifacts (empty until you train)
â”œâ”€â”€ data-forge/             # data pipeline: raw public datasets -> model_data/
â”œâ”€â”€ training/                # trains the Sketch tier + Z-Image-Turbo;
â”‚                              deprecated Planner/Critic training kept for reference
â”œâ”€â”€ inference/                # SwapOrchestrator + real backends + FastAPI service
â”œâ”€â”€ inference-runtime/         # CLI harness: run a real agentic session for testing
â”œâ”€â”€ inference-frontend/         # working Node.js control panel â€” Setup/Install tab
â”‚                                  drives the installer below; Studio tab is a full
â”‚                                  session UI (chat, finalize, critique, live GPU/
â”‚                                  RAM utilization, pixel-forming render preview)
â”œâ”€â”€ src/                        # superseded by inference-frontend/ â€” see src/README.md
â”œâ”€â”€ setup.sh                    # root orchestrator: phased environment setup
â”œâ”€â”€ run_data_forge.sh            # root orchestrator: data pipeline
â”œâ”€â”€ train.sh                      # root orchestrator: training, all four tiers
â”œâ”€â”€ run_inference.sh               # root orchestrator: the FastAPI service
â””â”€â”€ pytest.ini                     # root-level: makes `pytest tests/` work from here
```

See `docs/architecture/DIRECTORY_LAYOUT.md` for exactly why it's split
this way and the full oldâ†’new package mapping (this was restructured
from two separate repos â€” `data-forge` and `krisna-orchestrator` â€” with
the latter further split along a training/inference boundary).

## The real end-to-end path, verified stage by stage

This section exists because every step below was checked against the
actual code before being written down here, not assumed from a script's
own comment â€” see `docs/review/21_frontend_and_scripts_merge.md` for the
verification trail.

**1. Data pipeline** (`./run_data_forge.sh run`) â€” writes to
`$DATA_ROOT/model_data/` (`DATA_ROOT` defaults to `/data_krisna` on
Linux; override with the `DATA_ROOT` env var). See `data-forge/README.md`.

**2. Training** (`./train.sh <tier>`, or `--list` to see all four with
their real design intent and current hyperparameters) â€” each tier's
launcher prints exactly where its checkpoint lands and what to export
next once it finishes:
- Sketch Stage 1 â†’ Stage 2 â†’ `checkpoints/sketch_stage2_512/checkpoint_final.pt`
- Z-Image-Turbo dreambooth base â†’ `checkpoints/polish_default_lora/`
- Z-Image-Turbo Diffusion-DPO refinement (on top of the base) â†’
  `models/dpo_checkpoints/<run>/final/`
- The VQGAN decoder (`./scripts/training/download_vqgan.sh`) â€” needed
  by **both** data-forge's Sketch-tier encoding **and** inference's
  VQ-token-to-pixel decode step at Finalize. Not a per-tier training
  artifact; download it once, independent of which tiers you train.

**3. Installing for inference** (`python3 scripts/inference/download_weights.py`,
or drive it from `inference-frontend/`'s Setup tab) â€” downloads the four
frozen HF models, auto-discovers your trained Sketch checkpoint and
Polish LoRA at the exact paths step 2 produces above, downloads the
VQGAN decoder automatically if `download_vqgan.sh` hasn't already been
run, and writes `.env.inference` at the repo root with every env var the
service needs (`KRISNA_SKETCH_CHECKPOINT`, `KRISNA_POLISH_DEFAULT_LORA_PATH`,
`KRISNA_VQGAN_CHECKPOINT`, `KRISNA_VQGAN_CONFIG`, `KRISNA_USE_REAL_BACKENDS=1`,
and more). This step is what makes "training â†’ inference" a real,
connected path rather than two halves you have to wire together by hand.

**4. Running inference** (`./run_inference.sh`, or
`KRISNA_USE_REAL_BACKENDS=1 ./run_inference.sh` once step 3 is done) â€”
starts the real FastAPI service. `source .env.inference` first (or let
`inference-frontend/`'s own launcher do it â€” see below).

**5. Using it** â€” either directly against the API (`POST /session`,
`POST /session/{id}/message`, `POST /session/{id}/finalize`,
`POST /session/{id}/critique`, `GET /session/{id}/render` for the actual
rendered image bytes, `GET /orchestrator/status` for live VRAM/RAM â€”
both declared-budget and real hardware readings), or through
`inference-frontend/`'s Studio tab, which is a working UI over exactly
these endpoints: chat with the Planner, watch a phase-pipeline diagram
track Plannerâ†’Sketchâ†’Polishâ†’Critic, finalize and watch the pixel-forming
canvas reveal the real render, critique it, and watch live GPU/RAM
utilization (both the admission ledger's declared budget and the actual
hardware probe) the whole time.

**Yes, this is connected end to end** â€” verified by tracing every
producer/consumer pair above against the real code and configs, not
assumed from the scripts' own comments.

## Quick start

```bash
# One-time setup (Python venvs + Node frontend deps)
./setup.sh                        # base + backends + sketch + polish + critic + verifiers
./setup.sh frontend                # optional â€” the Node control panel (npm install)

# Data pipeline
./run_data_forge.sh run --dry-run

# Training (see what each tier actually needs first)
./train.sh --list

# Install for inference (downloads weights, locates your trained
# checkpoints, writes .env.inference) â€” real backends, needs a GPU
python3 scripts/inference/download_weights.py --dry-run   # check first
python3 scripts/inference/download_weights.py

# Inference service
./run_inference.sh                                        # MockBackend â€” no GPU needed
set -a; source .env.inference; set +a; ./run_inference.sh  # real backends

# Control panel (optional â€” a UI over the same API; brings the real
# backend up itself if .env.inference already exists)
cd inference-frontend && npm start

# Test a real agentic session against MockBackend from the CLI instead
python inference-runtime/run_agentic_session.py --finalize --critique

# Run every test in the monorepo in one pass (356 tests, no GPU required
# â€” GPU-dependent tests are gated behind pytest.importorskip("torch")
# and skip cleanly if it's absent)
pytest tests/
```

## Where to go next

| I want to... | Go to |
|---|---|
| Understand the data pipeline (sources, licenses, preprocessing) | `data-forge/README.md`, `docs/data-forge/` |
| Train the Sketch tier or Z-Image-Turbo | `training/README.md` |
| Understand the swap orchestrator, backends, or low-VRAM mode | `inference/README.md` |
| Use the working control-panel UI | `inference-frontend/README.md` |
| Manually test a trained checkpoint from the CLI | `inference-runtime/README.md` |
| Understand the data-forge â†” training sync contract | `docs/architecture/SYNC_DESIGN.md` |
| See exactly what was researched/verified and why each decision was made, with citations | `docs/architecture/RESEARCH_AND_CITATIONS.md` |
| See the full, phase-by-phase design-sync audit (21 phases and counting) â€” every bug found, every fix verified against real code, every citation checked against its primary source | `docs/review/README.md` |
| See the full PRD-vs-implementation reasoning trail | `docs/architecture/`, each package's README's "what's genuinely still missing" sections |

## Test suite

356 tests across the three Python packages (`data_forge`,
`krisna_training`, `krisna_inference`), runnable together from the root:

```bash
pytest tests/                    # everything
pytest tests/data_forge/          # data pipeline only
pytest tests/training/             # training only
pytest tests/inference/             # inference/orchestrator only
```

No GPU required for any of the above â€” GPU-dependent tests use
`pytest.importorskip("torch")` and skip cleanly if it's absent, rather
than aborting collection for the whole run (fixed a real inconsistency
in one test file that used a bare `import torch` instead â€” see
`docs/review/21_frontend_and_scripts_merge.md`).
#   D a t a - F o r g e   â ¬    Z e r o - T o u c h   D a t a   P i p e l i n e   ( v 1 6 )  
  
 >   A u t o m a t e d   d a t a   p i p e l i n e   f o r   t h e   * * K r i s n a * *   p r o j e c t   â ¬    a   c o n v e r s a t i o n a l ,   a g e n t i c  
 >   U I - d e s i g n   s y s t e m .   T h i s   r e v i s i o n   i m p l e m e n t s   t h e   P R D ' s   * * n o - R L H F - l o o p * *   d e c i s i o n :  
 >   e v e r y   t r a i n i n g - d a t a   s o u r c e   i s   e i t h e r   a   r e a l ,   h u m a n - l a b e l e d   p u b l i c   d a t a s e t ,   o r   a  
 >   m o d e l   t h a t   s h i p s   * * f r o z e n * *   a n d   n e e d s   n o   t r a i n i n g   d a t a   f r o m   t h i s   p i p e l i n e   a t   a l l .  
 >   T h e r e   i s   n o   A I - j u d g e / s e l f - d i s t i l l a t i o n   s t e p   a n y w h e r e   i n   t h i s   r e p o s i t o r y .  
 >  
 >   O f   t h e   f i v e   c o m p o n e n t s   i n   t h e   o r i g i n a l   P R D   m o d e l   s t a c k ,   t h i s   p i p e l i n e   t r a i n s  
 >   * * t w o * * :   t h e   S k e t c h   T i e r   ( f r o m   s c r a t c h )   a n d   Z - I m a g e - T u r b o   ( f i n e - t u n e   +  
 >   D i f f u s i o n - D P O   a l i g n m e n t ) .   T h e   P l a n n e r   ( Q w e n 3 . 5 - 9 B ) ,   Q w e n - I m a g e - E d i t - 2 5 1 1 ,   a n d  
 >   t h e   G e m m a - 4   C r i t i c   a l l   s h i p   f r o z e n   â ¬    s e e   " W h a t   T h i s   P i p e l i n e   D o e s   N O T   T r a i n "  
 >   b e l o w .   S e e   ` . . / d o c s / d a t a - f o r g e / D A T A _ S O U R C E S . m d `   f o r   t h e   f u l l   s o u r c e   r e g i s t r y   a n d  
 >   ` . . / d o c s / d a t a - f o r g e / D A T A _ C O M P L E T E N E S S . m d `   /   ` . . / d o c s / d a t a - f o r g e / A R C H I T E C T U R E . m d `   f o r   t h e   r e a s o n i n g   t r a i l  
 >   b e h i n d   t h i s   r e v i s i o n .  
  
 # #   W h a t   t h i s   p i p e l i n e   a c t u a l l y   p r o d u c e s  
  
 |   O u t p u t   |   F e e d s   |   D a t a   s o u r c e   |  
 | - - - | - - - | - - - |  
 |   ` m o d e l _ d a t a / s k e t c h _ t i e r _ m a s k g i t / `   |   S k e t c h   T i e r ,   t r a i n e d   f r o m   s c r a t c h   |   R I C O ,   C L A Y ,   E n r i c o ,   W e b U I   ( r e a l   U I   s c r e e n s h o t s )   |  
 |   ` m o d e l _ d a t a / p o l i s h _ z i m a g e _ t u r b o / `   |   Z - I m a g e - T u r b o ,   L o R A / Q L o R A   f i n e - t u n e   |   P D 1 2 M ,   C C 1 2 M   +   t h e   U I   s o u r c e s   a b o v e   |  
 |   ` m o d e l _ d a t a / d p o _ a l i g n m e n t / g e n e r a l / `   |   Z - I m a g e - T u r b o ,   D i f f u s i o n - D P O   S t a g e   1   |   P i c k - a - P i c   v 2 ,   H P D v 2   ( r e a l   h u m a n   p r e f e r e n c e   p a i r s )   |  
 |   ` m o d e l _ d a t a / d p o _ a l i g n m e n t / d o m a i n / `   |   Z - I m a g e - T u r b o ,   D i f f u s i o n - D P O   S t a g e   2   |   D e s i g n S e n s e - 1 0 k ,   D e s i g n P r e f   ( r e a l   h u m a n - d e s i g n e r   p r e f e r e n c e   p a i r s   â ¬    * * n o t   y e t   p u b l i c l y   r e l e a s e d * * ,   s e e   b e l o w )   |  
 |   ` m o d e l _ d a t a / p l a n n e r _ r a g _ c o r p u s / `   |   P r o d u c t - s i d e   R A G   i n d e x   ( n o t   a   t r a i n i n g   s e t )   |   U I C r i t   ( r e a l   h u m a n   c r i t i q u e   t e x t )   |  
 |   ` m o d e l _ d a t a / e v a l _ e x t e r n a l / `   |   H e l d - o u t   e v a l u a t i o n   o n l y   |   T A S T E ,   P a r t i P r o m p t s   â ¬    s t r u c t u r a l l y   u n r e a c h a b l e   f r o m   a n y   t r a i n i n g   p a t h   |  
  
 # #   W h a t   t h i s   p i p e l i n e   d o e s   N O T   t r a i n  
  
 -   * * P l a n n e r   ( Q w e n 3 . 5 - 9 B ) * *   s h i p s   f r o z e n .   I t   u s e s   R A G   o v e r   U I C r i t ' s   r e a l   c r i t i q u e  
     t e x t   ( ` m o d e l _ d a t a / p l a n n e r _ r a g _ c o r p u s / ` )   p l u s   c o n s t r a i n e d   J S O N   d e c o d i n g   a t  
     i n f e r e n c e   t i m e ,   i n   t h e   p r o d u c t ,   n o t   h e r e .  
 -   * * Q w e n - I m a g e - E d i t - 2 5 1 1 * *   s h i p s   f r o z e n .   T h e   " p o l i s h ,   q u a l i t y   p a t h "   u s e s  
     z e r o - s h o t   i n - c o n t e x t - l e a r n i n g   e d i t   c o n d i t i o n i n g   +   S D E d i t - s t y l e   p a r t i a l  
     d e n o i s i n g   a t   i n f e r e n c e   t i m e   â ¬    n o   p a i r e d   e d i t - t r a i n i n g   d a t a   i s   g e n e r a t e d   o r  
     n e e d e d .  
 -   * * G e m m a - 4   C r i t i c * *   s h i p s   f r o z e n   a s   a n   o n - d e m a n d ,   p r o d u c t - s i d e   c r i t i q u e  
     f e a t u r e .   I t   i s   n e v e r   t r a i n e d   b y   t h i s   p i p e l i n e ,   a n d   i t s   o u t p u t   i s   n e v e r   u s e d  
     a s   t r a i n i n g   d a t a   f o r   a n y t h i n g   e l s e   â ¬    t h e r e   i s   n o   A I - j u d g e - l a b e l e d   d a t a  
     a n y w h e r e   i n   t h i s   r e p o s i t o r y .  
  
 I f   y o u ' r e   l o o k i n g   f o r   a   ` s 0 7 _ 5 _ e d i t _ p a i r s ` ,   ` s 0 1 _ 6 _ p l a n n e r _ s y n t h e s i s ` ,   o r  
 ` s 1 0 _ 5 _ c r i t i c _ p r e f e r e n c e `   s t a g e   f r o m   a n   e a r l i e r   r e v i s i o n :   t h e y   w e r e   r e m o v e d ,  
 n o t   r e n a m e d .   T h e   t r a i n i n g   t a s k s   t h e y   f e d   n o   l o n g e r   e x i s t .  
  
 # #   A r c h i t e c t u r e  
  
 ` ` `  
 O r c h e s t r a t o r   ( C h u n k - B a s e d )  
     S t a g e   0     M a n i f e s t   P l a n n i n g                 - -   I n i t   D B ,   s t o r a g e   c h e c k ,   w a t c h e r   r e p o r t  
     S t a g e   1     F e t c h   +   L i c e n s e                     - -   T i e r - 1  
     S t a g e   1 . 5     U I C r i t   J o i n                         - -   - -  
     S t a g e   1 . 6     P r e f e r e n c e   P a i r s               - -   T i e r - 1   ( i n d e p e n d e n t   s t r e a m ,   s e e   b e l o w )  
     S t a g e   2     D e d u p   ( F A I S S )                         - -   C L I P  
     S t a g e   3     Q u a l i t y   S c o r i n g                     - -   T i e r - 1  
     S t a g e   3 . 5     P I I   S c r u b                             - -   M e d i a P i p e  
     S t a g e   4     S a f e t y                                       - -   T i e r - 1  
     S t a g e   4 . 5     E s c a l a t i o n                           - -   T i e r - 2  
     S t a g e   5     R e c a p t i o n                                 - -   T i e r - 1  
     S t a g e   5 - O C R     O C R   E n r i c h m e n t               - -   D e e p S e e k - O C R  
     S t a g e   5 . 5     P I I   T e x t   R e d a c t                 - -   R e g e x  
     S t a g e   6     S t r u c t u r e   E x t r a c t                 - -   T i e r - 1  
     S t a g e   7     R o u t i n g   &   S h a r d                     - -   - -  
     S t a g e   8     E n c o d i n g   ( Z - I m a g e   +   V Q )     - -   V A E s / V Q  
     S t a g e   8 . 5     D P O   E n c o d i n g   ( d i s a b l e d   b y   d e f a u l t )   - -   V A E   ( s e e   n o t e   b e l o w   â ¬    S t a g e   1 . 6 ' s   o u t p u t   i s   a c t u a l l y   c o n s u m e d   b y   t r a i n i n g / ' s   s y n c _ d p o _ p a i r s . p y ,   n o t   t h i s   s t a g e )  
     S t a g e   9     H e l d o u t   C a r v e                         - -   - -  
     S t a g e   1 0     A u d i t   P a s s                             - -   T i e r - 1   +   T i e r - 2  
     S t a g e   1 1     R e g i s t r y   W a t c h                     - -   - -  
     S t a g e   1 2     M o d e l   D a t a   E x p o r t               - -   - -  
 ` ` `  
  
 * * T w o   i n d e p e n d e n t   s t r e a m s ,   n o t   o n e   l i n e a r   p i p e l i n e . * *   T h e   m a i n   m a n i f e s t   s t r e a m  
 ( S t a g e   0   - >   1 0 )   p r o c e s s e s   s i n g l e   U I / g e n e r a l   i m a g e s   t h r o u g h   d e d u p ,   q u a l i t y ,  
 s a f e t y ,   c a p t i o n i n g ,   s t r u c t u r e   e x t r a c t i o n ,   a n d   e n c o d i n g   â ¬    t h i s   i s   w h a t   f e e d s  
 t h e   S k e t c h   T i e r   a n d   Z - I m a g e - T u r b o ' s   b a s e   f i n e - t u n e .   P r e f e r e n c e   p a i r s   ( P i c k - a -  
 P i c   v 2 ,   H P D v 2 ,   D e s i g n S e n s e - 1 0 k ,   D e s i g n P r e f )   a r e   a   * * s e p a r a t e   s t r e a m * * :   S t a g e  
 1 . 6   d e d u p s / b l u r s / s a f e t y - c l a s s i f i e s   t h e m   d i r e c t l y   ( t h e y   n e v e r   e n t e r   t h e  
 S Q L i t e   m a n i f e s t   â ¬    a   r a n k e d   p a i r   i s n ' t   a   s i n g l e - i m a g e   r e c o r d ) .   * * C o r r e c t i o n : * *  
 t h i s   u s e d   t o   s a y   S t a g e   8 . 5   " e n c o d e s   t h e m   i n t o   D P O   t r a i n i n g   l a t e n t s  
 a f t e r w a r d "   â ¬    t h a t   s t a g e   i s   d i s a b l e d   b y   d e f a u l t  
 ( ` c o n f i g s / p i p e l i n e . y a m l ` :   ` e n a b l e d :   f a l s e ` )   a n d   h a s   z e r o   r e a l   c o n s u m e r s ;  
 ` t r a i n i n g / d a t a _ f o r g e _ b r i d g e / s y n c _ d p o _ p a i r s . p y `   r e a d s   S t a g e   1 . 6 ' s   r a w   p a i r  
 i m a g e s   d i r e c t l y   v i a   t h e   s h a r e d   B l o b S t o r e ,   a n d   ` t r a i n _ d p o . p y `   l i v e - e n c o d e s  
 t h e m   a t   t r a i n   t i m e   i n s t e a d .   S e e   ` d o c s / d a t a - f o r g e / D A T A _ C O M P L E T E N E S S . m d `  
 a n d   ` s 0 8 _ 5 _ d p o _ e n c o d i n g . p y ` ' s   o w n   m o d u l e   d o c s t r i n g   f o r   t h e   f u l l   h i s t o r y .  
 S t a g e   1 2   p u l l s   f r o m   t h e   m a i n   s t r e a m   t o   b u i l d   ` m o d e l _ d a t a / ` ;   t h e  
 p r e f e r e n c e - p a i r   s t r e a m   i s   c o n s u m e d   d i r e c t l y   b y   t h e   t r a i n i n g   b r i d g e   a b o v e ,  
 n o t   t h r o u g h   S t a g e   1 2 .  
  
 ` 0 5 `   a n d   ` 0 5 - O C R `   a r e   t w o   d i s t i n c t ,   s e p a r a t e l y - r e g i s t e r e d   s t a g e s  
 ( ` s 0 5 _ r e c a p t i o n `   a n d   ` s 0 5 _ o c r _ e n r i c h m e n t ` ) .  
  
 # #   R e q u i r e m e n t s  
  
 -   * * O S * * :   U b u n t u   2 2 . 0 4   /   2 4 . 0 4   L T S   ( p r o d u c t i o n ) ,   W i n d o w s   ( d e v e l o p m e n t )  
 -   * * C U D A * * :   1 2 . 4   ( s t r i c t l y   p i n n e d )  
 -   * * P y t h o n * * :   3 . 1 0   o r   3 . 1 1  
 -   * * G P U * * :   4 8 G B   V R A M   ( e . g . ,   R T X   6 0 0 0   A d a ,   A 6 0 0 0 ,   L 4 0 S )  
 -   * * R A M * * :   1 2 8 G B   s y s t e m   R A M   r e c o m m e n d e d  
 -   * * D i s k * * :   > = 3 T B   a t   ` D A T A _ R O O T `  
  
 # #   S e t u p   &   V e r i f i c a t i o n  
  
 # # #   1 .   E n v i r o n m e n t   B o o t s t r a p  
 ` ` ` p o w e r s h e l l  
 c d   d a t a - f o r g e  
 S e t - E x e c u t i o n P o l i c y   - S c o p e   P r o c e s s   - E x e c u t i o n P o l i c y   B y p a s s  
 . \ s c r i p t s \ s e t u p _ e n v . p s 1  
 c o n d a   a c t i v a t e   k r i s n a - f o r g e  
 ` ` `  
 I n s t a l l s   P y t h o n   3 . 1 0 ,   ` t o r c h + c u 1 2 4 ` ,   ` f a i s s - g p u ` ,   ` p a n d a s ` / ` p y a r r o w `   ( n e e d e d   f o r  
 ` p r e f e r e n c e _ p a i r s / ` ,   ` u i _ c r i t i q u e / ` ,   a n d   D P O - l a t e n t   m e t a d a t a   o u t p u t ) ,   t h e n  
 ` p i p   i n s t a l l   - e   . [ d e v ] ` .  
  
 # # #   2 .   P r e - F l i g h t   S c h e m a   V e r i f i c a t i o n  
 ` ` ` p o w e r s h e l l  
 p y t h o n   s c r i p t s / v e r i f y _ s c h e m a s . p y  
 ` ` `  
  
 # # #   3 .   P i p e l i n e   D r y - R u n   V a l i d a t i o n  
 ` ` ` p o w e r s h e l l  
 $ e n v : D A T A _ R O O T = " D : \ k f _ d a t a "  
 d a t a - f o r g e   r u n   - - d r y - r u n  
 ` ` `  
  
 # #   U s a g e  
  
 ` ` ` p o w e r s h e l l  
 $ e n v : D A T A _ R O O T = " D : \ k f _ d a t a "  
 $ e n v : H F _ T O K E N = " y o u r _ h u g g i n g f a c e _ t o k e n "  
  
 #   S m o k e   t e s t   ( 1 0 0 - r e c o r d   m i c r o - b a t c h )  
 d a t a - f o r g e   r u n   - - c h u n k - s i z e   1 0 0   - - l i m i t   1 0 0  
  
 #   F u l l   p i p e l i n e   r u n  
 d a t a - f o r g e   r u n  
  
 #   R e s u m e   f r o m   c h e c k p o i n t  
 d a t a - f o r g e   r u n   - - r e s u m e  
  
 #   R u n   s p e c i f i c   s t a g e s  
 d a t a - f o r g e   r u n   - - s t a g e s   0 , 1 , 2  
  
 #   R u n   j u s t   t h e   p r e f e r e n c e - p a i r   p o s t - p r o c e s s i n g   ( d e d u p / b l u r / s a f e t y )  
 d a t a - f o r g e   r u n   - - s t a g e s   1 . 6  
  
 #   R u n   j u s t   t h e   D P O   l a t e n t   e n c o d i n g   ( d i s a b l e d   b y   d e f a u l t   â ¬    s e e   t h e   n o t e  
 #   a b o v e ;   o n l y   u s e f u l   i f   y o u ' v e   r e - e n a b l e d   i t   f o r   a   s p e c i f i c   r e a s o n )  
 d a t a - f o r g e   r u n   - - s t a g e s   8 . 5  
  
 #   R e g i s t r y   w a t c h e r   ( s c h e d u l e   v i a   T a s k   S c h e d u l e r )   â ¬    a l s o   w a t c h e s   f o r  
 #   D e s i g n S e n s e - 1 0 k / D e s i g n P r e f   p u b l i c   r e l e a s e s ,   s e e   c o n f i g s / d a t a s e t s . y a m l  
 d a t a - f o r g e   r e g i s t r y   c h e c k  
  
 d a t a - f o r g e   m a n i f e s t   s t a t s  
 d a t a - f o r g e   m a n i f e s t   q u e r y   - - s t a t u s   e x c l u d e d _ p e n d i n g _ r e v i e w  
 ` ` `  
  
 # #   C o n f i g u r a t i o n  
  
 A l l   c o n f i g u r a t i o n   l i v e s   i n   ` c o n f i g s / ` :  
 -   ` p i p e l i n e . y a m l `   â ¬    s t a g e   t o g g l e s ,   t h r e s h o l d s ,   p a t h s ,   c h u n k   s i z e s ,   s t o r a g e   e s t i m a t e s  
 -   ` m o d e l s . y a m l `   â ¬    p i n n e d   m o d e l   v e r s i o n s ,   q u a n t   s e t t i n g s ,   V R A M   b u d g e t s .   O n l y   ` z _ i m a g e _ v a e `   r e m a i n s   a s   a n   e n c o d e r   â ¬    n o   ` q w e n _ i m a g e _ v a e ` ,   n o   ` c r i t i c `   m o d e l   e n t r y   ( b o t h   r e m o v e d ;   t h o s e   m o d e l s   a r e   f r o z e n ,   d a t a - f o r g e   n e v e r   l o a d s   t h e m ) ,   a n d   ` m a s k g i t _ v q `   w a s   r e m o v e d   ( s y n c   a u d i t   i t e m   # 1 :   i t   n e v e r   h a d   a   w o r k i n g   i m p l e m e n t a t i o n ,   a n d   i t s   o n l y   c a l l e r   w a s   d e a d   c o d e   â ¬    s e e   ` d o c s / r e v i e w / 0 1 _ s k e t c h _ t i e r . m d ` )  
 -   ` d a t a s e t s . y a m l `   â ¬    1 5   r e g i s t e r e d   s o u r c e s ;   s e e   t h e   t a b l e   b e l o w  
  
 P a t h s   i n   ` p i p e l i n e . y a m l `   a r e   * * r e l a t i v e   t o   ` D A T A _ R O O T ` * * .  
  
 # #   S t a g e   R e f e r e n c e  
  
 |   S t a g e   |   N a m e   |   G P U   M o d e l   |   P u r p o s e   |  
 | - - - - - - - | - - - - - - | - - - - - - - - - - - | - - - - - - - - - |  
 |   0 0   |   M a n i f e s t   P l a n n i n g   |   - -   |   I n i t   D B ,   s t o r a g e   c h e c k   ( i m a g e - r e c o r d   a n d   p r e f e r e n c e - p a i r   b u d g e t s   t r a c k e d   s e p a r a t e l y ) ,   r e a d   w a t c h e r   r e p o r t   |  
 |   0 1   |   F e t c h   &   L i c e n s e   |   T i e r - 1   |   D o w n l o a d   +   i n l i n e   l i c e n s e   v e r i f i c a t i o n .   F i v e   d i s t i n c t   f e t c h   s h a p e s :   ` u r l _ l i s t `   ( P D 1 2 M / C C 1 2 M ) ,   ` h f _ p a r q u e t _ i m a g e s `   ( R I C O   â ¬    i m a g e s   e m b e d d e d   i n   p a r q u e t ,   n o t   l o o s e   f i l e s ) ,   ` c a p t i o n _ j o i n `   ( S c r e e n 2 W o r d s ) ,   ` p r e f e r e n c e _ p a i r ` / ` h p d v 2 _ r a n k e d _ l i s t `   ( D P O   s o u r c e s ) ,   ` e v a l _ r e f e r e n c e `   ( T A S T E / P a r t i P r o m p t s )   |  
 |   0 1 . 5   |   U I C r i t   J o i n   |   - -   |   J o i n s   U I C r i t ' s   r e a l   h u m a n   c r i t i q u e / r a t i n g s   o n t o   a l r e a d y - i n g e s t e d   R I C O   r e c o r d s .   F e e d s   t h e   P l a n n e r ' s   R A G   c o r p u s ,   n o t   a   t r a i n i n g   s e t   |  
 |   0 1 . 6   |   P r e f e r e n c e   P a i r s   |   T i e r - 1   |   C r o s s - s o u r c e   d e d u p   +   f a c e - b l u r   +   * * N S F W / h a r m f u l - c o n t e n t   s a f e t y   c l a s s i f i c a t i o n * *   o n   r e a l   D P O   p r e f e r e n c e   p a i r s .   R e f u s e s   t o   r u n   w i t h o u t   a   T i e r - 1   e n g i n e   r a t h e r   t h a n   s k i p   t h e   s a f e t y   c h e c k   |  
 |   0 2   |   D e d u p   |   C L I P   |   F A I S S   n e a r - d u p l i c a t e   r e m o v a l   ( m a i n   m a n i f e s t   s t r e a m   o n l y )   |  
 |   0 3   |   Q u a l i t y   |   T i e r - 1   |   A e s t h e t i c / r e s o l u t i o n   s c o r i n g   |  
 |   0 3 . 5   |   P I I   S c r u b   |   M e d i a P i p e   |   F a c e   b l u r ,   t e x t   r e d a c t i o n   ( s h a r e d   ` d a t a _ f o r g e / u t i l s / p i i _ f a c e s . p y ` ,   a l s o   u s e d   b y   S t a g e   1 . 6 )   |  
 |   0 4   |   S a f e t y   |   T i e r - 1   |   N S F W / h a r m f u l   c l a s s i f i c a t i o n   ( m a i n   m a n i f e s t   s t r e a m )   |  
 |   0 4 . 5   |   E s c a l a t i o n   |   T i e r - 2   |   B o r d e r l i n e   s e c o n d   o p i n i o n   |  
 |   0 5   |   R e c a p t i o n   |   T i e r - 1   |   D e n s e   s t r u c t u r a l   c a p t i o n i n g   |  
 |   0 5 - O C R   |   O C R   E n r i c h m e n t   |   D e e p S e e k - O C R   |   T e x t - i n - i m a g e   e x t r a c t i o n   |  
 |   0 5 . 5   |   P I I   T e x t   R e d a c t   |   R e g e x   |   R e d a c t s   t e x t   f o u n d   b y   t h e   O C R   p a s s   |  
 |   0 6   |   S t r u c t u r e   |   T i e r - 1   |   U I   l a y o u t   J S O N   e x t r a c t i o n   |  
 |   0 7   |   R o u t i n g   |   - -   |   D o m a i n   t a g g i n g   +   s h a r d   a s s i g n m e n t   ( ` u i _ f i r s t `   v s   ` g e n e r a l _ d e s i g n ` )   |  
 |   0 8   |   E n c o d i n g   |   V A E s / V Q   |   Z - I m a g e - T u r b o   l a t e n t s   +   M a s k G I T   V Q   t o k e n s   ( ` u i _ f i r s t ` - o n l y )   +   c o n t r o l   m a p s .   * * N o   Q w e n - I m a g e - E d i t - 2 5 1 1   b r a n c h * *   â ¬    t h a t   m o d e l   i s   f r o z e n   |  
 |   0 8 . 5   |   D P O   E n c o d i n g   â ¬    * * d i s a b l e d   b y   d e f a u l t * *   |   V A E   |   W o u l d   e n c o d e   S t a g e   1 . 6 ' s   p r e f e r e n c e   p a i r s   i n t o   Z - I m a g e - T u r b o ' s   l a t e n t   s p a c e ;   ` t r a i n _ d p o . p y `   l i v e - e n c o d e s   i n s t e a d   a n d   n e v e r   r e a d s   t h i s   s t a g e ' s   o u t p u t   â ¬    s e e   t h e   n o t e   a b o v e   |  
 |   0 9   |   H e l d o u t   C a r v e   |   - -   |   S t r a t i f i e d   e v a l   s p l i t ;   r e j e c t s   e n c o d i n g - i n c o m p l e t e   r e c o r d s   f i r s t   |  
 |   1 0   |   A u d i t   P a s s   |   T i e r - 1   +   T i e r - 2   |   V L M - a s - j u d g e   o n   2 - 5 %   s a m p l e   |  
 |   1 1   |   R e g i s t r y   W a t c h   |   - -   |   M o d e l / d a t a s e t   r e l e a s e   p o l l i n g ,   i n c l u d i n g   e x p l i c i t   w a t c h e s   f o r   D e s i g n S e n s e - 1 0 k / D e s i g n P r e f   |  
 |   1 2   |   M o d e l   D a t a   E x p o r t   |   - -   |   S e g m e n t s   i n t o   ` m o d e l _ d a t a / ` :   ` s k e t c h _ t i e r _ m a s k g i t / ` ,   ` p o l i s h _ z i m a g e _ t u r b o / ` ,   ` d p o _ a l i g n m e n t / { g e n e r a l , d o m a i n } / ` ,   ` p l a n n e r _ r a g _ c o r p u s / ` ,   ` e v a l _ e x t e r n a l / `   |  
  
 # #   S o u r c e   R e g i s t r y   S u m m a r y  
  
 S e e   ` . . / d o c s / d a t a - f o r g e / D A T A _ S O U R C E S . m d `   f o r   f u l l   d e t a i l .   F i f t e e n   r e g i s t e r e d   s o u r c e s :  
  
 -   * * F o u n d a t i o n   ( r e a l   i m a g e s ,   t r a i n e d   o n   d i r e c t l y ) : * *   P D 1 2 M ,   C C 1 2 M ,   R I C O   ( c o r e   +   s e m a n t i c ) ,   C L A Y ,   E n r i c o ,   W e b U I ,   S c r e e n 2 W o r d s  
 -   * * D P O   a l i g n m e n t   ( r e a l   h u m a n   p r e f e r e n c e   p a i r s ) : * *   P i c k - a - P i c   v 2 ,   H P D v 2   ( g e n e r a l ) ;   D e s i g n S e n s e - 1 0 k ,   D e s i g n P r e f   ( U I / d e s i g n - d o m a i n   â ¬    * * D e s i g n P r e f   i s   c o n f i r m e d   n o t   y e t   p u b l i c l y   r e l e a s e d * * ;   D e s i g n S e n s e - 1 0 k   h a s   n o   c o n f i r m e d   p u b l i c   r e p o   a n d   a   c o n f i r m e d   C C   B Y - N C - N D   4 . 0   l i c e n s e   t h a t   w o u l d   n e e d   s e p a r a t e   l e g a l   s i g n - o f f   e v e n   o n c e   r e l e a s e d )  
 -   * * R A G   c o r p u s   ( n o t   t r a i n e d   o n ) : * *   U I C r i t  
 -   * * E v a l u a t i o n   o n l y ,   s t r u c t u r a l l y   u n r e a c h a b l e   f r o m   t r a i n i n g : * *   T A S T E ,   P a r t i P r o m p t s  
  
 # #   K n o w n   I s s u e s ,   F i x e d   T h i s   R e v i s i o n   ( v 1 6 )  
  
 -   * * R I C O   f e t c h   p r o d u c e d   z e r o   r e c o r d s . * *   ` r i c o _ c o r e ` / ` r i c o _ s e m a n t i c `   w e r e  
     c o n f i g u r e d   w i t h   ` f i l e _ p a t t e r n s :   [ " * . j p g " , " * . p n g " , " * . j s o n " ] `   a g a i n s t   t w o   l i v e  
     H F   r e p o s   t h a t   s h i p   i m a g e s   e m b e d d e d   a s   b y t e s   i n s i d e   p a r q u e t   f i l e s ,   n o t   l o o s e  
     f i l e s   â ¬    ` a l l o w _ p a t t e r n s `   m a t c h e d   n o t h i n g ,   s o   e v e r y   r u n   s i l e n t l y   f e t c h e d   z e r o  
     i m a g e s   f r o m   t h e   s i n g l e   m o s t   f o u n d a t i o n a l   d a t a s e t   i n   t h e   c o r p u s   ( C L A Y ,  
     E n r i c o ,   a n d   S c r e e n 2 W o r d s   a l l   j o i n   o n t o   R I C O   r e c o r d s   t h a t   w o u l d   n e v e r   h a v e  
     e x i s t e d ) .   N e w   ` d o w n l o a d _ m o d e :   " h f _ p a r q u e t _ i m a g e s " `   d e c o d e s   t h e   e m b e d d e d  
     i m a g e   c o l u m n   d i r e c t l y .   S e e   ` . . / t e s t s / d a t a _ f o r g e / t e s t _ f e t c h _ h f _ p a r q u e t _ i m a g e s . p y ` .  
 -   * * R I C O ' s   t w o   i m a g e - c o l u m n   n a m e s ,   f u l l y   c o n f i r m e d ,   n o t   a u t o - d e t e c t e d . * *  
     ` r i c o _ c o r e `   ( c r e a t i v e - g r a p h i c - d e s i g n / R i c o )   u s e s   ` s c r e e n s h o t ` ;   ` r i c o _ s e m a n t i c `  
     ( V o x e l 5 1 / r i c o )   u s e s   ` i m a g e `   â ¬    c o n f i r m e d   d i r e c t l y   a g a i n s t   e a c h   l i v e   d a t a s e t  
     c a r d   ( r i c o _ c o r e ' s   d o c u m e n t e d   f e a t u r e   s c h e m a ;   r i c o _ s e m a n t i c ' s   o w n   D a t a  
     S t u d i o   p r e v i e w   t a b l e   h e a d e r ) .   T h e   t w o   r e p o s   a r e   i n d e p e n d e n t l y   e x p o r t e d   a n d  
     d o   n o t   s h a r e   a   c o l u m n   n a m e   d e s p i t e   c o v e r i n g   t h e   s a m e   u n d e r l y i n g   s c r e e n s .  
     B o t h   a r e   h a r d c o d e d   i n   ` d a t a s e t s . y a m l `   n o w ,   n o t   l e f t   t o   r u n t i m e  
     a u t o - d e t e c t i o n .   ` r i c o _ s e m a n t i c ` ' s   l i c e n s e   i s   a l s o   n o w   c o n f i r m e d   C C   B Y   4 . 0  
     d i r e c t l y   f r o m   i t s   d a t a s e t   c a r d   ( w a s   a   s o f t e r   " b e l i e v e d "   c l a i m ) .  
 -   * * P r e f e r e n c e - p a i r   i m a g e s   w e r e   n e v e r   s c r e e n e d   f o r   N S F W / h a r m f u l   c o n t e n t . * *  
     S t a g e   1 . 6   p r e v i o u s l y   r a n   w i t h   ` e n g i n e = N o n e `   a n d   d i d   d e d u p / b l u r   o n l y .   N o w  
     r e q u i r e s   a   T i e r - 1   e n g i n e   a n d   r e f u s e s   t o   r u n   w i t h o u t   o n e ;   ` u n s a f e ` - t i e r   p a i r s  
     a r e   d r o p p e d ,   ` b o r d e r l i n e `   p a i r s   k e p t   b u t   f l a g g e d .   S e e  
     ` . . / t e s t s / d a t a _ f o r g e / t e s t _ p r e f e r e n c e _ p a i r s _ s a f e t y . p y ` .  
 -   * * H P D v 2 ' s   r e a l   s c h e m a   d i d n ' t   m a t c h   t h e   g e n e r i c   p r e f e r e n c e - p a i r   f e t c h   p a t h . * *  
     H P D v 2   s h i p s   a   v a r i a b l e - l e n g t h   r a n k e d - l i s t   f o r m a t   ( ` h u m a n _ p r e f e r e n c e :  
     l i s t [ i n t ] ` ,   ` f i l e _ p a t h :   l i s t [ s t r ] ` ) ,   n o t   a   f i x e d   t w o - i m a g e   t a b l e   â ¬    t h e  
     g e n e r i c   p a t h   w o u l d   h a v e   f o u n d   z e r o   u s a b l e   p a i r s .   D e d i c a t e d  
     ` d o w n l o a d _ m o d e :   " h p d v 2 _ r a n k e d _ l i s t " `   a d a p t e r   a d d e d .  
 -   * * T h e   H P D v 2   f i x   a b o v e   i n i t i a l l y   c a u s e d   a   s e c o n d - o r d e r   b u g * * ,   t w i c e :   t h e   n e w  
     ` " h p d v 2 _ r a n k e d _ l i s t " `   m o d e   w a s n ' t   r e c o g n i z e d   b y   S t a g e   1 . 6 ' s   s o u r c e   f i l t e r  
     ( w o u l d   h a v e   s k i p p e d   H P D v 2 ' s   p a i r s   e n t i r e l y )   o r   b y   t h e   s t o r a g e - p r o j e c t i o n  
     m e t h o d s   i n   ` c o n f i g . p y `   ( w o u l d   h a v e   m i s - p r o j e c t e d   H P D v 2   a s   o r d i n a r y   i m a g e  
     r e c o r d s   i n s t e a d   o f   p r e f e r e n c e   p a i r s ) .   C e n t r a l i z e d   i n t o   o n e  
     ` D a t a s e t S p e c . P R E F E R E N C E _ P A I R _ D O W N L O A D _ M O D E S `   c o n s t a n t   s o   t h e   t w o   c a n ' t   d r i f t  
     a p a r t   a g a i n .  
 -   * * C L A Y   a n d   P D 1 2 M ' s   l i c e n s e s   w e r e   m a r k e d   m o r e   c o n s e r v a t i v e l y   t h a n   t h e   f a c t s  
     s u p p o r t . * *   C L A Y   i s   C C   B Y   4 . 0   p e r   i t s   o w n   p a p e r ' s   c o p y r i g h t   l i n e   ( w a s   " u n c l e a r  
     a r c h i v e d   r e p o ,   l i k e l y   e x c l u d e d " ) .   P D 1 2 M ' s   C D L A - P e r m i s s i v e - 2 . 0   a n d   i t s  
     ` u r l ` / ` c a p t i o n `   c o l u m n   n a m e s   a r e   n o w   c o n f i r m e d   a g a i n s t   t h e   l i v e   d a t a s e t  
     c a r d ,   n o t   a n   u n v e r i f i e d   g u e s s .  
 -   * * S t o r a g e - a c c o u n t i n g   d o u b l e - c o u n t i n g . * *   ` S t o r a g e M a n a g e r . c a l c u l a t e _ p r o j e c t e d _  
     s i z e `   u s e d   t o   s u m   e v e r y   ` p e r _ r e c o r d _ e s t i m a t e s `   k e y   a g a i n s t   o n e   r e c o r d   c o u n t  
     â ¬    a d d i n g   p r e f e r e n c e - p a i r   s t o r a g e   e s t i m a t e s   w o u l d   h a v e   m u l t i p l i e d   a  
     m a t e r i a l l y   l a r g e r   p e r - p a i r   c o s t   b y   t h e   i m a g e - r e c o r d   c o u n t .   S p l i t   i n t o  
     d i s j o i n t   p e r - i m a g e   /   p e r - p r e f e r e n c e - p a i r   e s t i m a t e   s e t s   w i t h   s e p a r a t e   c o u n t s .  
 -   * * ` d a t a _ f o r g e / u t i l s / c o m p l e t e n e s s . p y `   r e q u i r e d   a n   a r t i f a c t   ( ` q w e n _ i m a g e _ l a t e n t ` )   t h a t  
     S t a g e   8   n o   l o n g e r   p r o d u c e s   f o r   a n y   r e c o r d * * ,   w h i c h   w o u l d   h a v e   f l a g g e d   1 0 0 %  
     o f   r e c o r d s   a s   e n c o d i n g - i n c o m p l e t e   a f t e r   t h e   Q w e n - I m a g e - E d i t - 2 5 1 1   b r a n c h   w a s  
     r e m o v e d .   F i x e d   t o   t h e   c u r r e n t   t w o - a r t i f a c t   ( ` z _ i m a g e _ l a t e n t ` ,   ` c o n t r o l _ m a p ` )  
     +   o n e   U I - o n l y   a r t i f a c t   ( ` v q _ t o k e n s ` )   r e q u i r e m e n t .  
 -   * * ` c l i . p y ` ' s   ` - - s t a g e s `   s h o r t c u t   m a p * *   ( ` 1 . 6 ` ,   ` 7 . 5 ` ,   ` 1 0 . 5 ` )   s t i l l   p o i n t e d  
     a t   d e l e t e d   s t a g e   n a m e s   a f t e r   t h e   n o - R L H F - l o o p   r e s t r u c t u r i n g   â ¬    f i x e d .  
 -   * * ` s 1 2 _ m o d e l _ d a t a _ e x p o r t . p y `   l i n k e d   p r o c e s s e d   a r t i f a c t s   b u t   n e v e r   t h e   r a w  
     i m a g e s   e i t h e r   d o w n s t r e a m   c o n s u m e r   a c t u a l l y   n e e d s . * *   F o u n d   w h i l e   w i r i n g   u p  
     k r i s n a - o r c h e s t r a t o r ' s   s y n c   b r i d g e  
     ( ` t r a i n i n g / d a t a _ f o r g e _ b r i d g e / ` ,   t h a t   p r o j e c t ' s   s i d e ) :   ` s k e t c h _ t i e r _ m a s k g i t / `  
     o n l y   l i n k e d   ` v q _ t o k e n s / `   ( O p e n - M A G V I T 2   ` . p t `   f i l e s   â ¬    t h a t   e n c o d e r ' s   o w n  
     l o a d i n g   c o d e   i n   ` e n g i n e . p y `   r a i s e s   ` R u n t i m e E r r o r `   o n   p u r p o s e ,   u n v e r i f i e d  
     w o r k i n g   w r a p p e r ) ,   a n d   ` p o l i s h _ z i m a g e _ t u r b o / `   o n l y   l i n k e d   ` l a t e n t s / `   ( n o  
     c o n s u m e r   â ¬    t h e   o f f i c i a l   ` t r a i n _ d r e a m b o o t h _ l o r a _ z _ i m a g e . p y `   s c r i p t   t a k e s  
     r a w   i m a g e s   a n d   c o m p u t e s   i t s   o w n   l a t e n t s   i n t e r n a l l y ) .   B o t h   e x p o r t e r s   n o w  
     a l s o   l i n k   ` i m a g e s / ` .   C a u g h t   a   s e c o n d ,   g e n u i n e   b u g   i n   t h e   s a m e   p a s s :   w i t h  
     z e r o   m a t c h i n g   r e c o r d s   f o r   a   d o m a i n ,   n e i t h e r   e x p o r t e r   c r e a t e d   i t s   o w n  
     o u t p u t   d i r e c t o r y   b e f o r e   w r i t i n g   ` c a p t i o n s . j s o n l ` ,   r a i s i n g  
     ` F i l e N o t F o u n d E r r o r `   â ¬    ` _ e x p o r t _ p l a n n e r _ r a g `   a l r e a d y   d i d   t h i s   c o r r e c t l y ,  
     t h e   o t h e r   t w o   d i d n ' t .   S e e   ` . . / t e s t s / d a t a _ f o r g e / t e s t _ s 1 2 _ i m a g e s _ e x p o r t . p y ` .  
  
 # #   S t i l l   O p e n   ( d o c u m e n t e d ,   n o t   h i d d e n )  
  
 -   * * D e s i g n S e n s e - 1 0 k   a n d   D e s i g n P r e f   h a v e   n o   f e t c h a b l e   p u b l i c   d a t a . * *   R e - c h e c k e d  
     d i r e c t l y   a g a i n s t   H u g g i n g F a c e   t w i c e   t h i s   r e v i s i o n ,   n o t   j u s t   c a r r i e d   f o r w a r d  
     f r o m   a n   e a r l i e r   p a s s .   D e s i g n P r e f   i s   * c o n f i r m e d *   n o t   y e t   r e l e a s e d   ( t w o  
     i n d e p e n d e n t   p a p e r s   s t a t e   t h i s   o u t r i g h t   â ¬    i t s   o w n   d e s c r i b i n g   p a p e r   a n d  
     T A S T E ' s   r e l a t e d - w o r k   s e c t i o n ) .   D e s i g n S e n s e - 1 0 k   h a s   n o   p u b l i c   r e p o   f o u n d   i n  
     e i t h e r   c h e c k ,   a n d   i t s   c o n f i r m e d   C C   B Y - N C - N D   4 . 0   l i c e n s e   w o u l d   n e e d   s e p a r a t e  
     l e g a l   s i g n - o f f   e v e n   o n c e   r e l e a s e d   ( N o D e r i v s   c o n f l i c t s   w i t h   t h i s   p i p e l i n e ' s  
     r e c a p t i o n   s t e p ;   N o n C o m m e r c i a l   i s   a   s e p a r a t e   c o n c e r n   f o r   a n y   p r o d u c t   t r a c k ) .  
     B o t h   ` r e p o _ i d :   n u l l `   i n   ` d a t a s e t s . y a m l ` ,   b o t h   t r a c k e d   v i a  
     ` w a t c h e r _ s c a n _ t a r g e t s `   f o r   a   f u t u r e   r e l e a s e .   * * P r a c t i c a l   e f f e c t : * *   Z - I m a g e -  
     T u r b o ' s   S t a g e - 2   ( U I - d o m a i n )   D P O   a l i g n m e n t   h a s   z e r o   u s a b l e   d a t a   u n t i l   o n e   o f  
     t h e s e   i s   r e l e a s e d   â ¬    S t a g e - 1   ( g e n e r a l ,   P i c k - a - P i c   v 2   +   H P D v 2 )   D P O   r u n s   a s  
     t h e   o n l y   a l i g n m e n t   s i g n a l   i n   t h e   m e a n t i m e .  
 -   * * G a m e L a b e l - 1 0 K ,   a   r e a l ,   A p a c h e - 2 . 0 ,   g e n e r a l - d o m a i n   p r e f e r e n c e   d a t a s e t ,  
     f o u n d   d u r i n g   t h i s   r e v i s i o n ' s   v e r i f i c a t i o n   b u t   n o t   y e t   w i r e d   i n * *   â ¬    i t s  
     f i l e   i s   a   s i n g l e   2 . 2 6 G B   C S V ,   n o t   p a r q u e t ,   s o   i t   d o e s n ' t   f i t   e i t h e r   e x i s t i n g  
     p r e f e r e n c e - p a i r   f e t c h   s h a p e ,   a n d   i t s   e x a c t   c o l u m n   n a m e s   w e r e n ' t  
     i n d e p e n d e n t l y   c o n f i r m e d   b e f o r e   t h i s   r e v i s i o n   c l o s e d .   S e e  
     ` . . / d o c s / d a t a - f o r g e / D A T A _ S O U R C E S . m d ` ' s   " V e t t e d   b u t   n o t   y e t   i n t e g r a t e d "   s e c t i o n   b e f o r e  
     a d d i n g   i t   â ¬    i n s p e c t   t h e   r e a l   f i l e   h e a d e r   f i r s t   r a t h e r   t h a n   g u e s s i n g   a  
     s c h e m a ,   s a m e   d i s c i p l i n e   t h a t   c a u g h t   t h e   P D 1 2 M / H P D v 2   n e a r - m i s s e s   a b o v e .  
  
 * * P r i o r   r e v i s i o n s   ( v 1 4 - v 1 5 ) ,   s t i l l   i n   e f f e c t : * *   O C R   e n r i c h m e n t ' s   m i s s i n g  
 c o n f i g   b l o c k ,   t h e   n e v e r - p u b l i s h e d   Q w e n - I m a g e - 2 . 0 - V A E   r e f e r e n c e ,   a   f u l l - t a b l e  
 m a n i f e s t   s c a n   r e p e a t e d   p e r   d a t a s e t ,   a n   a r b i t r a r y   W i n d o w s   M A X _ P A T H   t h r e s h o l d ,  
 d e a d   F A I S S   c o n f i g ,   a n   u n c l o s e d   ` h t t p x . A s y n c C l i e n t `   i n   t h e   r e g i s t r y   w a t c h e r ,  
 t h r e e   U I   d a t a s e t s   m i s s i n g   f r o m   t h e   d o m a i n   t a g g e r ,   a n d   t w o   d u p l i c a t e d  
 ` S t a g e R e s u l t `   c l a s s e s .  
  
 # #   T e s t i n g  
  
 ` ` ` b a s h  
 p y t e s t   t e s t s /   - v   - m   " n o t   i n t e g r a t i o n "             #   U n i t   t e s t s ,   n o   G P U   r e q u i r e d  
 p y t e s t   t e s t s /   - v   - m   i n t e g r a t i o n                         #   R e q u i r e s   G P U   +   m o d e l s   d o w n l o a d e d  
 p y t e s t   t e s t s /   - - c o v = d a t a _ f o r g e   - - c o v - r e p o r t = h t m l  
 ` ` `  
  
 1 0 0 +   t e s t s ,   a l l   p a s s i n g   a s   o f   t h i s   r e v i s i o n   ( v e r i f i e d   v i a   ` p y _ c o m p i l e `   a c r o s s  
 e v e r y   m o d u l e ,   a   f u l l   ` l o a d _ c o n f i g ( ) `   p a s s   a g a i n s t   a l l   t h r e e   Y A M L   f i l e s ,   t h e  
 o r c h e s t r a t o r ' s   s t a g e - r e g i s t r a t i o n / o r d e r i n g   v a l i d a t o r ,   a n d   a   f r e s h   z i p  
 e x t r a c t i o n   r e - r u n   â ¬    n o t   j u s t   ` p y t e s t ` ' s   o w n   r e p o r t ) .  
 