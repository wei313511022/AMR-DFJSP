# LaTeX, from zero

This folder is the working source for an anonymous IEEE RA-L initial submission.
Compile after every substantive edit and resolve the first reported error before
continuing.

---

## 1. What LaTeX actually is

You write a **plain text file** (`.tex`) that describes your document. A program
(`pdflatex`) reads it and produces a PDF. You never see the PDF while typing — you
compile, then look.

Three ideas cover 95% of what you need:

| you write | you get |
|---|---|
| `\command{argument}` | does something to the argument |
| `\begin{name} ... \end{name}` | an *environment* — a block with special behaviour |
| `% text` | a comment, ignored |

That's it. `\textbf{hello}` makes **hello** bold. `\begin{equation} x=1 \end{equation}`
makes a numbered equation.

---

## 2. Install it (once, ~30 min)

**Windows:** install **MiKTeX** from miktex.org. Accept "install missing packages on
the fly = Yes". Then install **TeXstudio** (texstudio.org) as the editor.

**Alternative with zero install:** **Overleaf** (overleaf.com). Free account, works in
the browser, compiles for you. Upload the `paper/` folder as a project. If you have any
doubt, start here — you can move to a local install later.

RA-L initial submissions use `ieeeconf.cls`, not the final journal layout. Download
the current `ieeeconf.zip` pack from the official RA-L Information for Authors page
and upload the class file with the project if your TeX installation does not provide
it. The document class in `main.tex` follows the required initial-submission line:
`\documentclass[letterpaper,10pt,conference]{ieeeconf}`.

---

## 3. Your files

```
paper/
  main.tex                  <- the master file; you compile THIS one
  introduction.tex          <- Section I
  related_work.tex          <- Section II
  problem_formulation.tex   <- Section III
  references.bib            <- bibliography database
  fig_layout.pdf            <- vector layout figure
  HOW_TO_LATEX.md           <- this file
```

`main.tex` is the only file you compile. It pulls in the others with `\input{...}`.
This is why the formulation lives in its own file — you can work on one section
without scrolling through the whole paper.

---

## 4. Compiling

**In TeXstudio:** open `main.tex`, press **F5**. Done.

**In Overleaf:** press the green **Recompile** button.

**On the command line**, from inside `paper/`:

```
latexmk -pdf main.tex
```

`latexmk` figures out how many passes are needed. If you don't have it, run these four
commands in this exact order:

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

**Why four times?** LaTeX makes one pass to find out what your references and citations
are, `bibtex` builds the bibliography, and the last two passes fill in the numbers.
Skip them and you get `??` where citation numbers should be. This is normal and not a
sign anything is wrong.

---

## 5. The five things you'll actually do

### Add a citation

Your `.bib` file has entries with a **key** as the first field:

```bibtex
@inproceedings{zhang2020l2d,
  author = {Zhang, Cong and Song, Wen and ...},
  title  = {Learning to Dispatch for Job Shop Scheduling...},
  ...
}
```

To cite it, write `\cite{zhang2020l2d}` in your text. Multiple at once:
`\cite{zhang2020l2d,song2023fjsp}`. Numbering is automatic and in order of appearance.

To add a new reference, get the BibTeX entry from Google Scholar (click the `"` icon
under a result → BibTeX) and paste it into `references.bib`.

### Add a section

```latex
\section{Experiments}
\label{sec:experiments}
Your text here.
```

The `\label` gives it a name you can refer to. Then anywhere else,
`Section~\ref{sec:experiments}` prints "Section VII" and updates itself if you reorder.
The `~` is a non-breaking space so "Section" and "VII" never split across lines.

### Write an equation

Numbered, referable:

```latex
\begin{equation}
\eta = m / |\mathcal{D}^{\mathrm{in}}| .
\label{eq:eta}
\end{equation}
```

Refer to it with `\eqref{eq:eta}`, which prints "(1)".

Inline maths goes between dollar signs: `the ratio $\eta$ is 3.2`.

Useful symbols: `\le \ge \neq \in \subset \sum \max \min \forall \alpha \Delta`.
Subscript `x_1`, superscript `x^2`, both `x_1^2`. More than one character needs braces:
`x_{\max}`.

### Add a figure

Put `myplot.pdf` in the `paper/` folder, then:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{myplot}
  \caption{Execution penalty against contention ratio.}
  \label{fig:penalty}
\end{figure}
```

Note: no file extension in `\includegraphics`. `[t]` means "put it at the top of a
page". Use `\columnwidth` for one column, `\textwidth` inside `figure*` for full width.
Prefer PDF figures over PNG — they stay sharp.

### Add a table

```latex
\begin{table}[t]
\caption{Execution penalty by fleet size}
\label{tab:penalty}
\centering
\begin{tabular}{@{}lrr@{}}
\toprule
Fleet & Penalty & Unroutable \\
\midrule
8  & 7.4\%  & 0.00 \\
16 & 14.9\% & 0.00 \\
\bottomrule
\end{tabular}
\end{table}
```

`&` separates columns, `\\` ends a row. `{lrr}` means left, right, right alignment —
one letter per column. **Percent signs must be escaped as `\%`** because `%` starts a
comment; forgetting this silently eats the rest of the line.

---

## 6. IEEE two-column specifics

Your paper is two narrow columns, which causes the one problem you'll hit repeatedly:
**things that are too wide.**

- A wide table or figure: use `table*` / `figure*` instead of `table` / `figure`. These
  span both columns but can normally appear only at the top of a page.
- A wide equation: split it. I did this twice in your formulation — see `eq:ideal`,
  where I introduced $\tilde{C}_k$ as an intermediate so the line fits.
- The warning is `Overfull \hbox (34pt too wide)`. Inspect every overfull box in the
  rendered PDF; even a small excess can be visible in a narrow IEEE column.

---

## 7. When it breaks

LaTeX errors are cryptic. Four causes account for nearly all of them.

**`Undefined control sequence`** — you used a command that doesn't exist, usually a
typo (`\textbf` vs `\textbold`) or a missing package. The line number in the error is
usually right.

**`File 'X.cls' not found`** — a missing package. MiKTeX offers to install it; say yes.
On Overleaf, everything is already there.

**`Missing $ inserted`** — you used a maths symbol outside maths mode. `\alpha` on its
own fails; `$\alpha$` works.

**`??` in the PDF instead of a number** — you didn't compile enough times, or you
referred to a `\label` that doesn't exist. Run the full four-pass sequence.

**General rule:** scroll to the *first* error and fix only that one. Later errors are
usually knock-on effects and disappear on their own.

If the build gets stuck in a strange state, delete the junk files (`.aux`, `.log`,
`.bbl`, `.blg`, `.out`, `.toc`) and compile fresh. Those are all regenerated; only
`.tex`, `.bib` and your figures matter.

---

## 8. Current paper status

The abstract, introduction, related work, problem formulation, policy architecture,
surrogate calibration, PPO training, experiments, limitations, and conclusion are all
included. Run `build-paper.cmd` after editing; the current clean build is eight pages and
has no undefined references or overfull boxes.

Before submission, confirm all of the following: the paper is anonymous; the eight-page
RA-L budget and any extra-page charges are acceptable; every result claim has a traceable
number and uncertainty; all citations resolve; no author identity or drafting note
remains; and PaperCept's PDF checker accepts the file.
