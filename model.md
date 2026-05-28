# TFT MuZero Model Architecture

This document outlines the architecture for the TFT MuZero agent, with a current focus on the **BoardGenerator** head.

## 1. BoardGenerator Architecture

The BoardGenerator is responsible for determining the optimal placement of units on the board given the current state and champion availability.

### 1.1 Inputs (Champion Availability Encoding)

The input to the BoardGenerator is a condensed availability vector based on the first 3304 features of the observation (board, bench, chosen state, etc.).

*   **Objective:** The model needs to know which champions are available and which are not.
*   **Encoding Scheme:** All values are normalized between 0 and 1.
    *   **Champion ID:** Represented implicitly by its position in the list (`list_position`).
    *   **Level (Star Level):** A single value normalized to `level / 3` (scaling from 0.33 to 1.0).
    *   **Chosen:** A single boolean value (1.0 for true, 0.0 for false).
*   **Vector Size:** `(1 + 1) * N` (where N is the total number of distinct champions in the simulator). This vector summarizes the overall champion availability to be passed into the generator.

### 1.2 Hidden Dimension & Architecture (Decoder Design)

*   **Architecture Type:** Since the BoardGenerator outputs a grid representation (the hex board), it acts as a decoder. We will use a network of **ConvTranspose** layers (or standard convolutions on an upsampled grid) to convert the 1D availability vector into the final 2D board state.
*   **Channel vs. Spatial Scaling:** Drawing from famous image generation models (like DCGAN generators or VAE decoders), the internal sizes scale in opposite directions:
    *   **Channel Dimension (Hidden State):** The number of channels/feature maps starts **large** (equal to or greater than the input dimension to fit original data + future predictions) and gradually **decreases** as it approaches the final output layer.
    *   **Spatial Dimension (Grid Size):** The 1D input is projected and reshaped into a small spatial bottleneck. ConvTranspose layers then **increase** (upsample) this spatial size step-by-step until reaching the final TFT board resolution (e.g., $4 \times 7$ hexes per player).

#### Proposed 3-Step Architecture for 4x7 Board
To reach the standard $4 \times 7$ TFT board size, we propose a 3-layer generator design:
1. **Linear Projection:** A dense layer maps the 1D availability vector to a flat tensor that is reshaped into a $1 \times 2$ spatial grid (e.g., shape `[512, 1, 2]`).
2. **ConvTranspose2d Layer 1:** Upsamples the grid by 2x. Using `kernel_size=(2,4)`, `stride=(2,2)`, and `padding=(0,1)` results in an intermediate grid of $2 \times 4$. The channels are halved (e.g., from 512 to 256).
3. **ConvTranspose2d Layer 2 (Output):** Upsamples the grid again. Using `kernel_size=(2,3)`, `stride=(2,2)`, and `padding=(0,1)` results in the final $4 \times 7$ spatial output. The channels drop to match the required number of output classes.

### 1.3 Outputs

*   **Probability Distribution:** The direct output of the BoardGenerator is a probability distribution for each unit in each hex across the final spatial grid.
*   **Action Translation:** The raw distribution is not an action. A separate module within the `muzero_agent` will ingest this probability distribution and compute the discrete actions (placements/movements) needed to achieve that board state.

*(Note: The rest of the models, outside the BoardGenerator, can utilize standard MLPs for now).*
