# Implementation Spec 02: HTR-VT with ABINet Language Branch

## 1. Architecture Redesign

We are upgrading `HTR_VT` from a pure CTC model to a **Hybrid CTC + Attention** model with explicit Language Modeling.

### Data Flow

1. **Input**: Image `(B, 1, 64, 512)`
2. **Backbone**: ResNet + ViT Encoder (Existing) -> Output Features `(B, 128, 768)`
3. **Branch 1 (CTC)**: Features -> Linear -> Logits `(B, 128, Num_Class)` -> **CTC Loss**
4. **Branch 2 (ABINet)**:
   - **Visual Decoding**: Features -> `PositionAttention` -> Visual Logits `(B, Max_Len, Num_Class)`
   - **Language Refinement**: Visual Logits (Softmax) -> `BCN` (Language Model) -> Language Features
   - **Fusion**: Visual Features + Language Features -> `FusionGate` -> Final Logits
   - **Loss**: **CrossEntropy Loss**

## 2. New Modules (Ported from ABINet)

Create a new file `model/abinet_layers.py` to house the ported code.

- **PositionAttention**: Converts variable length visual features to fixed length character queries.
- **BCN (Bidirectional Cloze Network)**: The core Language Model.
- **GatedFusion**: Combines visual and language signals.

## 3. Label Conversion

- **CTC Converter**: Already exists (for Branch 1).
- **Attn Converter**: Need a new converter for Branch 2 that outputs **fixed-length padded tensors** (e.g., shape `[B, 26]`) with `<GO>` and `<EOS>` tokens.

## 4. Loss Function

$$\mathcal{L}_{total} = \lambda_{ctc} \mathcal{L}_{CTC} + \lambda_{attn} \mathcal{L}_{CE}(v_{logits}) + \lambda_{lang} \mathcal{L}_{CE}(f_{logits})$$

- Recommended weights: $\lambda_{ctc}=1.0, \lambda_{attn}=1.0, \lambda_{lang}=1.0$.
