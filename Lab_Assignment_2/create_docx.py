from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import os

doc = Document()

style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)

for level in range(1, 4):
    hs = doc.styles[f'Heading {level}']
    hs.font.name = 'Times New Roman'
    hs.font.color.rgb = RGBColor(0, 0, 0)

BASE = '/Users/raj/CUB/Sem2/NNDL/Lab_Assignment_2'

# ── Title ──
title = doc.add_heading('Lab Assignment 2: Training Deep Neural Networks', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in title.runs:
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor(0, 0, 0)

doc.add_paragraph()  # spacer

# ════════════════════════════════════════════
#  METHODS
# ════════════════════════════════════════════
doc.add_heading('Methods', level=1)

doc.add_heading('Dataset and Preprocessing', level=2)
doc.add_paragraph(
    'I chose CIFAR-10, which contains 60,000 color images (32\u00d732\u00d73) across 10 classes '
    '(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), split into '
    '50,000 training and 10,000 test samples. This dataset suits our needs because natural-image '
    'classification with small networks is genuinely hard \u2014 pixel-level features vary with '
    'lighting, pose, and background, making it unlikely that a shallow model would exceed 50% '
    'accuracy. For preprocessing, each image was converted to a float tensor and channel-wise '
    'normalized to zero mean and unit variance using the known CIFAR-10 statistics '
    '(means: 0.4914, 0.4822, 0.4465; standard deviations: 0.2470, 0.2435, 0.2616). '
    'This normalization is especially important when using sigmoid activations, since '
    'unnormalized pixel values push pre-activations into the sigmoid\u2019s saturating tails from the start.'
)
doc.add_paragraph(
    'To verify dataset difficulty, I built a two-layer fully-connected network: the flattened '
    '3072-dimensional input feeds into a 256-unit hidden layer with sigmoid activation, followed '
    'by a 10-way softmax output. It was trained for 10 epochs with vanilla SGD (lr=0.01, batch size=1).'
)

doc.add_heading('Baseline Deep Network', level=2)
doc.add_paragraph(
    'The baseline is a 5-layer convolutional network tailored to CIFAR-10\u2019s image modality. '
    'Layer 1 is a Conv2d (3\u219216 channels, 3\u00d73, padding 1) followed by sigmoid and MaxPool(2\u00d72), '
    'producing 16\u00d716\u00d716 feature maps. Layer 2 is Conv2d (16\u219232, 3\u00d73, padding 1) with sigmoid '
    'and MaxPool(2\u00d72), giving 32\u00d78\u00d78. Layer 3 is Conv2d (32\u219264, 3\u00d73, padding 1) with sigmoid '
    'and MaxPool(2\u00d72), giving 64\u00d74\u00d74. The features are flattened to 1024 dimensions and passed '
    'through Layer 4, a Linear(1024\u2192128) with sigmoid, and finally Layer 5, a Linear(128\u219210) '
    'producing logits. Softmax is applied inside the cross-entropy loss computation using the '
    'log-sum-exp trick for numerical stability. Training used vanilla SGD (lr=0.01, batch size=1) '
    'for 10 epochs, with the optimizer implemented from scratch \u2014 no torch.optim was used. '
    'Gradients were computed via autograd and parameters updated manually.'
)

doc.add_heading('Activation Functions', level=2)
doc.add_paragraph(
    'I selected Leaky ReLU and SiLU as the two replacement activations. '
    'Leaky ReLU, f(x) = max(x, 0.1x), is the standard fix for dying neurons and fully avoids '
    'sigmoid\u2019s saturation \u2014 its derivative is always either 1.0 or 0.1, so gradients never vanish. '
    'SiLU (Swish), f(x) = x \u00b7 sigmoid(x), is a smoother alternative that behaves like ReLU for '
    'large positive inputs but has a slight non-monotonic dip for negative inputs. Both were '
    'implemented from scratch using basic tensor operations (torch.where for Leaky ReLU, '
    'element-wise multiply with custom sigmoid for SiLU). The modified models have the identical '
    'architecture as the baseline \u2014 only the activation function is swapped.'
)
p = doc.add_paragraph()
run = p.add_run('Hypothesis: ')
run.bold = True
p.add_run(
    'I expected both activations to substantially outperform sigmoid, with Leaky ReLU having an '
    'edge because its gradient floor of 0.1 never saturates. Sigmoid\u2019s derivative maxes out at '
    '0.25 and approaches zero in the tails, so after five layers the gradient signal gets '
    'multiplied by roughly 0.25\u2074 \u2248 0.004. SiLU uses sigmoid internally, so I anticipated it '
    'might show some instability under noisy batch-size-1 training.'
)

doc.add_heading('Mini-Batch SGD', level=2)
doc.add_paragraph(
    'After identifying the best activation function, I retrained that model with mini-batch SGD '
    'at batch sizes of 16, 32, and 64. The implementation averages the per-sample losses within '
    'each batch before calling backward \u2014 since the gradient of the average equals the average '
    'of the gradients (linearity of differentiation), this is mathematically equivalent to '
    'averaging per-sample gradients. The learning rate (0.01) and epoch count (10) were held '
    'constant so batch size is the only variable.'
)

doc.add_heading('Mini-Batch SGD with Momentum', level=2)
doc.add_paragraph(
    'Momentum maintains a running average of past gradients: m\u1d62\u207a\u00b9 = \u03b1\u00b7m\u1d62 + g\u1d62, '
    'then \u03b8\u1d62\u207a\u00b9 = \u03b8\u1d62 \u2212 lr\u00b7m\u1d62\u207a\u00b9, where g\u1d62 is the current mini-batch gradient '
    'and m is initialized to zero. I tested \u03b1 = 0.5, 0.9, and 0.99 using the best batch size '
    'from the previous experiment. All other hyperparameters stayed fixed.'
)

doc.add_heading('Extended Architecture', level=2)
doc.add_paragraph(
    'To set up the skip-connection experiments, I added 10 Conv2d(16, 16, 3\u00d73, padding=1) layers '
    'immediately after the first conv+pool block. These layers all operate on 16-channel, 16\u00d716 '
    'feature maps and do not change tensor shape. Each is followed by the best activation function '
    '(Leaky ReLU). This produces a 15-layer network total. Training used mini-batch SGD with '
    'momentum using the best batch size (16) and best momentum (\u03b1=0.5) for 10 epochs. During '
    'the first epoch, I recorded the L1-norm of every parameter\u2019s gradient at each step, then '
    'averaged across steps to get one gradient magnitude per parameter.'
)
p = doc.add_paragraph()
run = p.add_run('Hypothesis: ')
run.bold = True
p.add_run(
    'I expected the extended model to perform worse than the 5-layer baseline. Ten extra layers '
    'mean the gradient passes through ten additional nonlinearities, compounding any attenuation '
    'even with Leaky ReLU.'
)

doc.add_heading('Skip Connection Configurations', level=2)
doc.add_paragraph(
    'I tested two skip-connection layouts, each with exactly 3 additive skip connections among '
    'the 10 extra layers:'
)
p = doc.add_paragraph()
run = p.add_run('Config A (length-2 skips): ')
run.bold = True
p.add_run(
    'Shortcuts across layers 0\u20131, 3\u20134, and 6\u20137. This creates frequent short-range gradient '
    'highways through every other pair of layers.'
)
p = doc.add_paragraph()
run = p.add_run('Config B (length-3 skips): ')
run.bold = True
p.add_run(
    'Shortcuts across layers 0\u20132, 3\u20135, and 6\u20138. Fewer but longer skip paths that provide '
    'broader gradient coverage.'
)
doc.add_paragraph(
    'Both use element-wise addition (tensors are the same shape). All other training details '
    'are identical to the extended model.'
)

# ════════════════════════════════════════════
#  RESULTS
# ════════════════════════════════════════════
doc.add_heading('Results', level=1)

doc.add_heading('Performance Summary', level=2)

# ── Main results table ──
rows_main = [
    ('Shallow 2-Layer (Sigmoid)',          '0.3958', '0.4409', '0.3958'),
    ('Deep Baseline (Sigmoid, bs=1)',      '0.5451', '0.5453', '0.5451'),
    ('Deep CNN (Leaky ReLU, bs=1)',        '0.6529', '0.6636', '0.6529'),
    ('Deep CNN (SiLU, bs=1)',              '0.3974', '0.4656', '0.3974'),
    ('Leaky ReLU, bs=16',                 '0.7283', '0.7385', '0.7283'),
    ('Leaky ReLU, bs=32',                 '0.6977', '0.7000', '0.6977'),
    ('Leaky ReLU, bs=64',                 '0.5792', '0.6381', '0.5792'),
    ('Leaky ReLU, bs=16, mom=0.5',        '0.7285', '0.7317', '0.7285'),
    ('Leaky ReLU, bs=16, mom=0.9',        '0.6875', '0.7017', '0.6875'),
    ('Leaky ReLU, bs=16, mom=0.99',       '0.1000', '0.0100', '0.1000'),
    ('Extended 15-layer (no skips)',       '0.1000', '0.0100', '0.1000'),
    ('Skip Config A (len-2 skips)',        '0.7099', '0.7094', '0.7099'),
    ('Skip Config B (len-3 skips)',        '0.7324', '0.7378', '0.7324'),
]

table = doc.add_table(rows=1 + len(rows_main), cols=4, style='Light Grid Accent 1')
table.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = table.rows[0].cells
for i, txt in enumerate(['Model', 'Test Accuracy', 'Avg Precision', 'Avg Recall']):
    hdr[i].text = txt
    for p in hdr[i].paragraphs:
        for r in p.runs:
            r.bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
for idx, (model, acc, prec, rec) in enumerate(rows_main):
    row = table.rows[idx + 1].cells
    row[0].text = model
    row[1].text = acc
    row[2].text = prec
    row[3].text = rec
    for c in range(1, 4):
        row[c].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()  # spacer

# ── Figures ──
figures = [
    ('fig1_loss_activations.png',  'Figure 1: Training loss curves \u2014 sigmoid baseline vs Leaky ReLU vs SiLU'),
    ('fig2_acc_activations.png',   'Figure 2: Test accuracy over epochs \u2014 sigmoid baseline vs Leaky ReLU vs SiLU'),
    ('fig3_batch_sizes.png',       'Figure 3: Test accuracy comparison for batch sizes 16, 32, 64'),
    ('fig4_momentum.png',          'Figure 4: Test accuracy comparison for momentum \u03b1 = 0.5, 0.9, 0.99'),
    ('fig5_skip_connections.png',  'Figure 5: Test accuracy \u2014 extended (no skips) vs Config A vs Config B'),
    ('fig6_gradient_norms.png',    'Figure 6: Per-layer L1 gradient norms for extended, Skip-A, and Skip-B'),
    ('fig8_summary.png',           'Figure 7: Final accuracy bar chart across all models'),
]

for fname, caption in figures:
    fpath = os.path.join(BASE, 'plots', fname)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(fpath, width=Inches(5.5))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(10)
    doc.add_paragraph()  # spacer

# ════════════════════════════════════════════
#  ANALYSIS
# ════════════════════════════════════════════
doc.add_heading('Analysis', level=1)

doc.add_heading('Trend 1: Activation function choice is the single biggest factor in deep-network training', level=2)
doc.add_paragraph(
    'The deep sigmoid baseline was stuck at 10% accuracy (random guessing) for the first 5 epochs '
    'before finally learning, reaching 54.5% by epoch 10. Leaky ReLU hit 56.9% after just one '
    'epoch and finished at 65.3%. This is the vanishing gradient problem in action \u2014 sigmoid\u2019s '
    'derivative maxes out at 0.25 and drops toward zero in the tails, so after five layers the '
    'gradient signal to the early conv layers was essentially zero for the first half of training. '
    'Leaky ReLU avoids this entirely since its derivative is always at least 0.1. SiLU started '
    'strong (60.9% by epoch 2) but then diverged \u2014 its loss increased from epoch 4 onward and '
    'accuracy collapsed to 39.7%. The likely cause is SiLU\u2019s non-monotonic region combining with '
    'noisy batch-size-1 gradients to create unstable updates. This showed that a theoretically '
    'fancier activation function is not always better; stability under noisy conditions matters '
    'just as much.'
)

doc.add_heading('Trend 2: Mini-batching had a bigger practical effect than momentum', level=2)
doc.add_paragraph(
    'Switching from batch size 1 to 16 jumped accuracy from 65.3% to 72.8% \u2014 a massive gain '
    'from just averaging 16 gradients together. Batch size 16 outperformed 32 (69.8%) and 64 '
    '(57.9%) because, at a fixed learning rate of 0.01, more frequent updates (3125 per epoch '
    'for bs=16 vs 781 for bs=64) mattered more than cleaner gradients. The bs=64 model\u2019s loss '
    'was still dropping at epoch 10, suggesting it was simply under-trained. Best batch size: 16. '
    'Momentum was surprisingly less impactful: \u03b1=0.5 gave 72.9% (nearly identical to no momentum '
    'at 72.8%), \u03b1=0.9 dropped to 68.8%, and \u03b1=0.99 completely diverged (NaN loss, 10% accuracy). '
    'With lr=0.01, high momentum builds up too much velocity and overshoots. Best \u03b1: 0.5.'
)

doc.add_heading('Trend 3: Skip connections revived a completely dead deep network', level=2)
doc.add_paragraph(
    'The extended 15-layer model without skips scored 10% \u2014 it never learned anything. Even with '
    'Leaky ReLU, passing gradients through 14 nonlinearities means worst-case scaling of '
    '0.1\u00b9\u2074 \u2248 10\u207b\u00b9\u2074 for early layers. Adding just three skip connections brought the model '
    'back to life: Config A (length-2 skips) reached 71.0% and Config B (length-3 skips) reached '
    '73.2%. The gradient norm plots confirm this \u2014 in the no-skip model, early-layer gradients '
    'were orders of magnitude smaller than later layers, while skip connections made gradient '
    'magnitudes more uniform across layers. Config B outperformed Config A because its three '
    'longer skips cover layers 0\u20138 (broader coverage), while Config A only covers layers 0\u20131, '
    '3\u20134, 6\u20137 leaving layers 2, 5, 8, 9 outside any skip path. Best skip configuration: '
    'Config B (length-3).'
)
doc.add_paragraph(
    'The shallow 2-layer model\u2019s accuracy of 39.6% (well under 50%) confirms CIFAR-10 is too '
    'hard for shallow models, validating the dataset choice.'
)

# ════════════════════════════════════════════
#  EXTRA CREDIT
# ════════════════════════════════════════════
doc.add_heading('Extra Credit', level=1)

doc.add_heading('Weight Decay', level=2)

rows_wd = [
    ('DeepCNN + WD \u03b2=0.0001',  '0.7363', '0.7402', '0.7363'),
    ('DeepCNN + WD \u03b2=0.001',   '0.7539', '0.7606', '0.7539'),
    ('DeepCNN + WD \u03b2=0.01',    '0.6947', '0.7111', '0.6947'),
    ('Skip-B + WD \u03b2=0.0001',   '0.7371', '0.7371', '0.7371'),
    ('Skip-B + WD \u03b2=0.001',    '0.7232', '0.7416', '0.7232'),
    ('Skip-B + WD \u03b2=0.01',     '0.6066', '0.6798', '0.6066'),
]

table2 = doc.add_table(rows=1 + len(rows_wd), cols=4, style='Light Grid Accent 1')
table2.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr2 = table2.rows[0].cells
for i, txt in enumerate(['Model', 'Test Accuracy', 'Avg Precision', 'Avg Recall']):
    hdr2[i].text = txt
    for p in hdr2[i].paragraphs:
        for r in p.runs:
            r.bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
for idx, (model, acc, prec, rec) in enumerate(rows_wd):
    row = table2.rows[idx + 1].cells
    row[0].text = model
    row[1].text = acc
    row[2].text = prec
    row[3].text = rec
    for c in range(1, 4):
        row[c].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()

# Weight decay figure
fpath = os.path.join(BASE, 'plots', 'fig7_weight_decay.png')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run()
run.add_picture(fpath, width=Inches(5.5))
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = cap.add_run('Figure 8: Weight decay comparison for DeepCNN and Skip-B at different \u03b2 values')
run.italic = True
run.font.size = Pt(10)
doc.add_paragraph()

doc.add_paragraph(
    'Weight decay adds a penalty term (\u03b2\u00b7\u03b8) to the momentum update, pushing weights toward '
    'zero to discourage memorization. For the DeepCNN (no skips), \u03b2=0.001 produced 75.4% \u2014 the '
    'highest accuracy of any model in this study and a clear improvement over the same model '
    'without decay (72.9%). \u03b2=0.0001 gave 73.6% (marginal help), while \u03b2=0.01 degraded to '
    '69.5% (too aggressive). For Skip-B, weight decay was less helpful: \u03b2=0.0001 gave 73.7% '
    '(roughly matching the no-decay 73.2%), \u03b2=0.001 dropped to 72.3%, and \u03b2=0.01 fell to '
    '60.7%. The skip-connection model was more sensitive because the additive identity paths '
    'amplify the effect of the weight penalty \u2014 shrinking weights also weakens the skip '
    'connections\u2019 ability to propagate signal. Best \u03b2: 0.001 for DeepCNN, 0.0001 for Skip-B. '
    'Moderate weight decay acts as an effective regularizer for simpler architectures, but models '
    'with skip connections already have a built-in regularizing effect from the identity paths, '
    'so additional decay can hurt.'
)

# ── Save ──
out_path = os.path.join(BASE, 'Lab2_Report.docx')
doc.save(out_path)
print(f'Saved to {out_path}')
