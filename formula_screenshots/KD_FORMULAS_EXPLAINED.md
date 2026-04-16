# Knowledge Distillation Loss Functions -- Detailed Explanation

> **Setting:** Teacher (all-MiniLM-L6-v2, frozen) distills into Student (TinyBERT-4L, trainable) for code search on the MBPP dataset. Both are bi-encoders that independently embed queries and documents.

---

## 1. Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\mathbf{S}_q,\ \mathbf{S}_d$ | Student query / document embedding matrices (B x D) |
| $\mathbf{T}_q,\ \mathbf{T}_d$ | Teacher query / document embedding matrices (B x D) |
| $s^s_{ij}$ | Student similarity score: cosine similarity between student query $i$ and document $j$ |
| $s^t_{ij}$ | Teacher similarity score: cosine similarity between teacher query $i$ and document $j$ |
| $B$ | Batch size (each batch has B query-document pairs; other pairs in the batch serve as in-batch negatives) |
| $\tau$ | Temperature for contrastive cross-entropy (default 0.05, sharpens the softmax) |
| $\tau_d$ | Distillation temperature (default 0.2, controls softness of teacher distributions) |
| $\lambda_d,\ \lambda_a,\ \lambda_p$ | Weighting coefficients for distillation / alignment / pairwise loss terms |
| $\sigma(\cdot)$ | Sigmoid function: $\sigma(x) = 1/(1 + e^{-x})$, maps any real number to (0, 1) |
| $\lVert \cdot \rVert_2$ | L2 (Euclidean) norm -- measures vector distance |

---

## 2. Shared Base Losses

Every method builds on one or both of these. Understanding them is key to understanding everything else.

### 2a. Contrastive Cross-Entropy Loss ($\mathcal{L}_{CE}$)

$$\mathcal{L}_{CE} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{B}\exp(s_{ij}/\tau)}$$

**Term-by-term breakdown:**

- $s_{ii}$: The similarity score between query $i$ and its **correct** (positive) document $i$. This is the diagonal of the score matrix.
- $s_{ij}$: The similarity score between query $i$ and document $j$ (for $j \neq i$, these are **negatives** -- other documents in the same batch that are not the correct answer).
- $\exp(s_{ii}/\tau)$: Exponentiated positive score, divided by temperature. Low $\tau$ (0.05) makes this sharply peaked -- the model must be very confident.
- $\sum_j \exp(s_{ij}/\tau)$: Sum over all documents in the batch (positive + negatives). This is the normalization denominator.
- The fraction is a softmax probability: "what fraction of total similarity mass goes to the correct document?"
- The $-\log$ turns this probability into a loss: probability 1.0 gives loss 0; lower probabilities give higher loss.
- $\frac{1}{B}$: Average over all queries in the batch.

**Intuition:** This is the standard InfoNCE / in-batch negatives loss. For each query, treat its paired document as the positive and all other documents in the batch as negatives. The model is trained to score the correct document highest. No teacher is involved -- this is pure supervised learning from labeled pairs.

**Role in the system:** Every single method uses this as the foundation. It ensures the student can do basic retrieval even without any teacher signal.

---

### 2b. Score Distribution Distillation ($\mathcal{L}_{KL}$)

$$\mathcal{L}_{KL} = \mathrm{KL}\!\left(\mathrm{softmax}\!\left(\frac{\mathbf{S}_s}{\tau_d}\right) \ \Big\|\ \mathrm{softmax}\!\left(\frac{\mathbf{S}_t}{\tau_d}\right)\right) \cdot \tau_d^{2}$$

**Term-by-term breakdown:**

- $\mathbf{S}_s$: The student's B x B score matrix. Entry $(i,j)$ is how similar the student thinks query $i$ is to document $j$.
- $\mathbf{S}_t$: The teacher's B x B score matrix. Same structure, but from the teacher's perspective.
- $\mathrm{softmax}(\mathbf{S}_t / \tau_d)$: The teacher's "soft labels." By dividing by $\tau_d$ (0.2) before softmax, we get a softer distribution than with $\tau$ (0.05). This reveals the teacher's nuanced ranking: instead of just "document 3 is best," it says "document 3 is best, document 7 is second-best, document 1 is third..." These inter-document relationships are the "dark knowledge" the teacher transfers.
- $\mathrm{softmax}(\mathbf{S}_s / \tau_d)$: The student's distribution over documents, at the same soft temperature.
- $\mathrm{KL}(\cdot \| \cdot)$: KL divergence measures how different the student's distribution is from the teacher's. It is zero when they match perfectly, and positive otherwise.
- $\tau_d^2$: Gradient magnitude correction. When temperature is used in the softmax, gradients are scaled down by $1/\tau_d^2$. Multiplying by $\tau_d^2$ compensates, keeping gradient magnitudes consistent across different temperature settings. (This is Hinton et al.'s standard trick from the original KD paper.)

**Intuition:** The teacher doesn't just know which document is correct -- it knows the relative ranking of all documents. A "hard" label says "document 3 is correct, everything else is wrong." A "soft" label from the teacher says "document 3 scores 0.8, document 7 scores 0.15, document 1 scores 0.04, ..." This is much richer supervision. The student learns to reproduce these nuanced rankings, not just the binary correct/incorrect signal.

**Important detail:** The score matrices here use **cross-encoder** style scoring: $\mathbf{S}_s = \mathbf{S}_q \cdot \mathbf{T}_d^\top$ (student queries against **teacher** documents) and $\mathbf{S}_t = \mathbf{T}_q \cdot \mathbf{T}_d^\top$ (teacher queries against teacher documents). This ensures the student is compared against the teacher in the teacher's own document space.

---

## 3. Method Variants -- From Simplest to Most Complex

### Method 1: Supervised Baseline

$$\mathcal{L} = \mathcal{L}_{CE}$$

**What it is:** No teacher involvement at all. The student learns purely from labeled query-document pairs using the contrastive loss. This is the control condition -- every other method should beat this, or the distillation is not helping.

**When you'd use this:** As a baseline to measure how much value the teacher adds.

---

### Method 2: Score Distillation

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_d \,\mathcal{L}_{KL}$$

**What it adds over supervised:** The $\mathcal{L}_{KL}$ term. Now the student gets two learning signals:
1. **From labels** ($\mathcal{L}_{CE}$): "document $i$ is correct for query $i$"
2. **From the teacher** ($\mathcal{L}_{KL}$): "here's how I'd rank ALL documents for each query"

**$\lambda_d$ (distill_weight):** Controls how much the student trusts the teacher vs. the labels. Higher $\lambda_d$ means "listen to the teacher more." In practice, this needs to be high (e.g., 50-100) because the $\tau_d^2$ factor makes the KL loss numerically small.

**Limitation:** The teacher only communicates through similarity scores -- it tells the student the ranking, but not *how* to achieve that ranking in embedding space. Two very different embedding geometries can produce the same ranking.

---

### Method 3: Embedding Distillation

$$\mathcal{L}_{align} = \frac{1}{B}\sum_{i=1}^{B} \left\|\mathbf{S}_{q_i} - \mathbf{T}_{q_i}\right\|_2$$

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_d \,\mathcal{L}_{KL} + \lambda_a \,\mathcal{L}_{align}$$

**The new term ($\mathcal{L}_{align}$) explained:**

- $\mathbf{S}_{q_i}$: Student's embedding vector for query $i$ (a point in D-dimensional space).
- $\mathbf{T}_{q_i}$: Teacher's embedding vector for the same query.
- $\lVert \mathbf{S}_{q_i} - \mathbf{T}_{q_i} \rVert_2$: Euclidean distance between them. This is zero when the student places its query embedding at exactly the same point as the teacher.
- $\frac{1}{B}\sum$: Average over all queries in the batch.

**Intuition:** Score distillation says "produce the same ranking." Embedding alignment says "produce the same vectors." This is a much stronger signal -- if your vectors match, your rankings will automatically match, but the reverse is not true. The student now inherits the teacher's entire geometric structure of the embedding space, not just its ranking behavior.

**Three learning signals combined:**
1. $\mathcal{L}_{CE}$: learn from labels
2. $\mathcal{L}_{KL}$: match the teacher's document rankings
3. $\mathcal{L}_{align}$: place query embeddings where the teacher places them

**Note:** This only aligns **query** embeddings. The student's document encoder receives no direct teacher supervision -- it only learns indirectly through the contrastive loss.

---

### Method 4: Hard-Negative Pairwise Distillation

$$\mathcal{L}_{pair} = \frac{1}{|M|}\sum_{(i,j)\in M} \mathrm{BCE}\!\left(\sigma\!\left(\frac{s^s_{ii} - s^s_{ij}}{\tau_d}\right),\ \sigma\!\left(\frac{s^t_{ii} - s^t_{ij}}{\tau_d}\right)\right)$$

$$M = \{(i,j) : j \in \text{top-}k\text{ hardest negatives for query } i\}$$

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_d \,\mathcal{L}_{KL} + \lambda_p \,\mathcal{L}_{pair}$$

**Term-by-term breakdown of $\mathcal{L}_{pair}$:**

- $s^s_{ii} - s^s_{ij}$: The student's **margin** -- how much higher it scores the positive document vs. negative document $j$. A large positive margin means the student is confident in its ranking.
- $s^t_{ii} - s^t_{ij}$: The teacher's margin for the same pair.
- $\sigma(\text{margin}/\tau_d)$: Sigmoid squashes the margin to a probability in (0, 1). This represents "how confident is the model that the positive beats this negative?" A value near 1 means very confident; near 0.5 means uncertain.
- $\mathrm{BCE}(\cdot, \cdot)$: Binary cross-entropy. The student's confidence (left) is trained to match the teacher's confidence (right) for each pair.
- $M$: The set of "hard negatives" -- for each query, the $k$ documents that the **teacher** scores highest (but aren't the correct document). These are the documents the teacher finds most confusable with the correct answer.
- $\frac{1}{|M|}$: Average over all hard-negative pairs.

**Intuition:** Regular score distillation (KL) treats all negatives equally. But most negatives are easy (completely unrelated documents) and carry little learning signal. This method asks the teacher: "which negatives do *you* find hardest to distinguish from the positive?" Then it focuses the student's learning specifically on those boundary cases. It's like a teacher saying: "Don't worry about the obviously wrong answers -- let me show you the tricky ones you'll likely confuse."

**Key difference from score distillation:** Score distillation matches the entire distribution over all B documents. Pairwise distillation zooms in on the top-$k$ hardest cases and matches the pairwise preference (positive vs. each hard negative) individually.

---

### Method 5: Query Embedding Alignment (QED)

$$\mathcal{L}_{align} = \frac{1}{B}\sum_{i=1}^{B} \left\|\mathbf{S}_{q_i} - \mathbf{T}_{q_i}\right\|_2$$

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_a \,\mathcal{L}_{align}$$

**How it differs from Method 3 (Embedding Distillation):** Same alignment loss, but **no score distillation** ($\mathcal{L}_{KL}$ is absent). The student gets:
1. Labels ($\mathcal{L}_{CE}$)
2. Query embedding targets ($\mathcal{L}_{align}$)

...but no ranking distribution from the teacher.

**Intuition:** This tests whether embedding alignment alone is enough, without the score distribution signal. It's the simplest possible embedding-based distillation approach.

---

### Method 6: Margin MSE

$$m^s_{ij} = s^s_{ii} - s^s_{ij} \qquad m^t_{ij} = s^t_{ii} - s^t_{ij}$$

$$\mathcal{L}_{mse} = \frac{1}{|M|}\sum_{\substack{i,j \\ i \neq j}} \left(m^s_{ij} - m^t_{ij}\right)^2$$

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_p \,\mathcal{L}_{mse}$$

**Term-by-term breakdown:**

- $m^s_{ij}$: Student's margin for query $i$ between its positive document and negative document $j$. If $m^s_{ij} = 0.3$, the student thinks the positive is 0.3 more similar than document $j$.
- $m^t_{ij}$: Teacher's margin for the same pair.
- $(m^s_{ij} - m^t_{ij})^2$: Squared difference between student and teacher margins. If the teacher has margin 0.5 and the student has margin 0.2, the loss is $(0.2 - 0.5)^2 = 0.09$.
- Sum over all pairs where $i \neq j$: every (positive, negative) combination in the batch.

**Intuition:** Rather than matching absolute scores (which can differ in scale between teacher and student) or full distributions (which are complex), this directly matches the **ranking gaps**. The teacher says: "I think the positive beats document 3 by a margin of 0.5 and beats document 7 by a margin of 0.1." The student is trained to reproduce those same margins.

**Key difference from Method 4 (Hard-Negative Pairwise):**
- Method 4 uses BCE on sigmoid-transformed margins (soft binary classification) and focuses only on the top-$k$ hardest negatives.
- Method 6 uses MSE on raw margins (regression) and considers **all** negatives equally.
- Method 4 also includes $\mathcal{L}_{KL}$; Method 6 does not.

---

### Method 7: BiMGA -- Bidirectional Margin-Guided Alignment (Ours)

$$m_i = s^t_{ii} - \max_{j \neq i}\, s^t_{ij} \qquad w_i = \sigma\!\left(\frac{m_i}{\tau_d}\right)$$

$$\mathcal{L}_{align} = \frac{1}{B}\sum_{i=1}^{B} w_i \left(\left\|\mathbf{S}_{q_i} - \mathbf{T}_{q_i}\right\|_2 + \left\|\mathbf{S}_{d_i} - \mathbf{T}_{d_i}\right\|_2\right)$$

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_d \,\mathcal{L}_{KL} + \lambda_a \,\mathcal{L}_{align}$$

**Term-by-term breakdown of the new components:**

**Margin-based confidence weight ($w_i$):**
- $s^t_{ii}$: Teacher's score for the correct (positive) document for query $i$.
- $\max_{j \neq i} s^t_{ij}$: Teacher's score for the **hardest** negative -- the non-positive document the teacher rates highest.
- $m_i = s^t_{ii} - \max_{j \neq i} s^t_{ij}$: The teacher's confidence margin. Large $m_i$ means the teacher easily distinguishes the positive from all negatives. Small $m_i$ means even the teacher finds this example ambiguous.
- $\sigma(m_i / \tau_d)$: Sigmoid maps the margin to a weight between 0 and 1. High-confidence examples ($m_i$ large) get $w_i \approx 1$; uncertain examples ($m_i$ small or negative) get $w_i \approx 0.5$ or lower.

**Bidirectional alignment:**
- $\lVert \mathbf{S}_{q_i} - \mathbf{T}_{q_i} \rVert_2$: Query alignment (same as Methods 3 and 5).
- $\lVert \mathbf{S}_{d_i} - \mathbf{T}_{d_i} \rVert_2$: **Document alignment** -- this is new. The student's document encoder is now directly supervised to place document embeddings where the teacher places them.
- These two terms are summed: the student is pulled toward the teacher on **both sides** of the encoder.

**Putting it together:** $w_i \cdot (\text{query distance} + \text{doc distance})$ means:
- When the teacher is confident ($w_i \approx 1$): "I'm sure about this example -- student, you should strongly align both your query and document embeddings with mine."
- When the teacher is uncertain ($w_i \approx 0.5$): "I'm not sure about this one -- don't trust my embeddings as strongly here, I might be wrong."

**Two key innovations over prior methods:**

1. **Bidirectional** (vs. query-only in Methods 3, 5): Prior methods only align query embeddings because they were designed for asymmetric architectures where the teacher's document encoder is reused at inference. In our symmetric setting, the student encodes documents itself at test time, so its document encoder also benefits from teacher guidance.

2. **Margin-guided weighting** (vs. uniform weighting): Prior alignment methods weight all examples equally. BiMGA down-weights examples where the teacher is uncertain, preventing the student from copying the teacher's mistakes on ambiguous cases.

---

### Method 8: BiMGA Uniform (Ablation)

$$\mathcal{L}_{align} = \frac{1}{B}\sum_{i=1}^{B} \left(\left\|\mathbf{S}_{q_i} - \mathbf{T}_{q_i}\right\|_2 + \left\|\mathbf{S}_{d_i} - \mathbf{T}_{d_i}\right\|_2\right)$$

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_d \,\mathcal{L}_{KL} + \lambda_a \,\mathcal{L}_{align}$$

**What changed:** The confidence weight $w_i$ is removed (equivalently, set to 1 for all examples). All examples contribute equally to the alignment loss.

**Purpose:** This is an ablation study to isolate the contribution of margin-guided weighting. By comparing BiMGA (Method 7) with BiMGA Uniform (Method 8), we can measure exactly how much the confidence weighting helps. The difference between them tells us whether selectively trusting the teacher on high-confidence examples is better than uniformly trusting the teacher on everything.

---

### Method 9: ADAM-Lite -- Dark Example Distillation

$$\tilde{\mathbf{d}}_k = \mathrm{normalize}\!\left(\alpha\,\mathbf{d}^{+} + (1-\alpha)\,\mathbf{d}^{-}_k\right)$$

$$c_i = \sigma\!\left(\frac{s^t_{i,+} - \overline{s^t_{i,\mathrm{dark}}}}{\tau_d}\right)$$

$$\mathcal{L}_{dark} = \frac{1}{B}\sum_{i=1}^{B} c_i \cdot \mathrm{KL}\!\left(\mathrm{softmax}\!\left(\frac{\tilde{\mathbf{S}}^s_i}{\tau_d}\right) \ \Big\|\ \mathrm{softmax}\!\left(\frac{\tilde{\mathbf{S}}^t_i}{\tau_d}\right)\right) \cdot \tau_d^{2}$$

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_d \left(\mathcal{L}_{KL} + \mathcal{L}_{dark}\right)$$

**Term-by-term breakdown:**

**Dark example generation ($\tilde{\mathbf{d}}_k$):**
- $\mathbf{d}^{+}$: The positive (correct) document embedding.
- $\mathbf{d}^{-}_k$: The $k$-th hard negative document embedding (selected by teacher scores).
- $\alpha$ (dark_mix_ratio, default 0.65): Interpolation weight. At $\alpha = 0.65$, dark examples are 65% positive and 35% negative -- they're *similar* to the correct document but corrupted with negative information.
- $\mathrm{normalize}(\cdot)$: L2 normalization ensures the synthetic document lives on the unit sphere (same as real embeddings).
- **Result:** $\tilde{\mathbf{d}}_k$ is a synthetic "dark" document that sits between the positive and negative in embedding space -- right at the decision boundary where discrimination is hardest.

**Confidence gate ($c_i$):**
- $s^t_{i,+}$: Teacher's score for the positive document.
- $\overline{s^t_{i,\text{dark}}}$: Teacher's average score across the dark examples.
- $c_i = \sigma(\text{gap}/\tau_d)$: If the teacher scores the positive much higher than the dark examples, $c_i \to 1$ (high confidence). If scores are similar, $c_i \to 0.5$ (low confidence).
- **Role:** High $c_i$ means the teacher can clearly distinguish the positive from the synthetic hard cases, so its distribution is trustworthy. Low $c_i$ means the dark examples successfully fooled even the teacher, so its signal is less reliable.

**Dark KL loss ($\mathcal{L}_{dark}$):**
- $\tilde{\mathbf{S}}^s_i$, $\tilde{\mathbf{S}}^t_i$: Score vectors over the set [positive doc, dark examples] from student and teacher respectively.
- The KL term is the same as $\mathcal{L}_{KL}$ but computed over synthetic documents instead of in-batch negatives.
- Weighted by $c_i$: the student focuses on dark examples where the teacher is confident.

**Intuition:** In-batch negatives are often easy (random documents from unrelated queries). This method manufactures adversarial near-miss documents that sit right at the decision boundary. It's like a teacher creating specifically challenging practice problems rather than assigning random ones. The confidence gate ensures the teacher only insists on these challenging examples when it's actually sure about the answer itself.

---

## 4. Comparison Table

### What Teacher Signal Does Each Method Use?

| Method | Labels | Score Distribution | Query Embeddings | Doc Embeddings | Confidence Weighting |
|--------|:------:|:-----------------:|:----------------:|:--------------:|:--------------------:|
| 1. Supervised | Yes | -- | -- | -- | -- |
| 2. Score Distill | Yes | KL divergence | -- | -- | -- |
| 3. Embed Distill | Yes | KL divergence | L2 alignment | -- | -- |
| 4. Hard-Neg Pairwise | Yes | KL divergence | -- | -- | via hard-neg selection |
| 5. QED Align | Yes | -- | L2 alignment | -- | -- |
| 6. Margin MSE | Yes | -- | -- | -- | -- |
| 7. **BiMGA** | Yes | KL divergence | L2 alignment | L2 alignment | margin-guided sigmoid |
| 8. BiMGA Uniform | Yes | KL divergence | L2 alignment | L2 alignment | -- (uniform) |
| 9. ADAM-Lite | Yes | KL divergence | -- | -- | confidence gate |

### How They Relate to Each Other

```
Supervised (baseline, no teacher)
  |
  +-- Score Distill (add teacher's ranking distribution)
  |     |
  |     +-- Embed Distill (add query embedding alignment)
  |     |     |
  |     |     +-- BiMGA (add doc alignment + margin weighting)
  |     |           |
  |     |           +-- BiMGA Uniform (ablation: remove margin weighting)
  |     |
  |     +-- Hard-Neg Pairwise (add focused pairwise preference on hardest cases)
  |     |
  |     +-- ADAM-Lite (add synthetic hard negatives with confidence gating)
  |
  +-- QED Align (query alignment only, no score KD)
  |
  +-- Margin MSE (match ranking margins, no distribution or embedding KD)
```

### Key Conceptual Axes

**Axis 1 -- What knowledge is transferred:**
- **Score-based** (Methods 2, 4, 6, 9): The teacher communicates through similarity scores. The student learns the teacher's ranking behavior but not its embedding geometry. Two very different vector spaces can produce identical rankings.
- **Embedding-based** (Methods 3, 5, 7, 8): The teacher communicates through its actual embedding vectors. The student inherits the teacher's geometric structure. This is strictly more information than score-based transfer.
- **Combined** (Methods 3, 7, 8): Use both signals simultaneously.

**Axis 2 -- What gets aligned:**
- **Query-only** (Methods 3, 5): Only the student's query encoder is pulled toward the teacher. The document encoder learns indirectly.
- **Bidirectional** (Methods 7, 8): Both the student's query AND document encoders are aligned with the teacher. This matters because at test time, the student encodes both sides.

**Axis 3 -- How examples are weighted:**
- **Uniform** (Methods 2, 3, 5, 6, 8): All examples contribute equally.
- **Confidence-weighted** (Methods 7, 9): Examples where the teacher is more confident contribute more to the loss. This prevents the student from copying the teacher's mistakes on ambiguous cases.
- **Hard-negative focused** (Method 4): Only the most confusable examples are used for pairwise learning.

---

## 5. Practical Intuition: A Teaching Analogy

Think of the teacher model as an expert tutor and the student model as a learner:

| Method | What the tutor does |
|--------|-------------------|
| **Supervised** | "Here's the answer key. Figure it out yourself." |
| **Score Distill** | "Here's how I'd rank all the options, including partial credit." |
| **Embed Distill** | "Here's my ranking AND here's my thought process (internal representation) for each question." |
| **Hard-Neg Pairwise** | "Let me show you the trick questions specifically -- here's how to tell apart the confusing options." |
| **QED Align** | "Just copy my thought process for the questions. Don't worry about my rankings." |
| **Margin MSE** | "Match how confident I am in each pairwise comparison." |
| **BiMGA** | "Copy my thought process for BOTH questions and answers, but only trust me when I'm confident." |
| **BiMGA Uniform** | "Copy my thought process for both questions and answers, trust me equally on everything." |
| **ADAM-Lite** | "Let me create custom practice problems at exactly your difficulty level, and I'll weight them by how sure I am of the answer." |
