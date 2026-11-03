# TCC - Fabricio Novicki

LaTeX structure for the Mechanical Engineering undergraduate thesis at UFSC, organized according to the university's 2024 TCC template and ABNT-oriented formatting.

## Compile

The project is configured for **pdfLaTeX** and uses Helvetica as a portable Arial-compatible sans-serif alternative.

Recommended:

```bash
latexmk -pdf main.tex
```

Or manually:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Project structure

- `main.tex`: document assembly and ordering;
- `metadata.tex`: author, title, advisor, committee and year;
- `config.tex`: typography, margins, spacing, captions and page numbering;
- `pretextual/`: cover, title page, approval page, abstract, lists etc.;
- `chapters/`: thesis chapters;
- `appendices/`: appendices and optional annexes;
- `figures/`: figures and UFSC logo extracted from the official template;
- `references.bib`: BibTeX database.

## Important conventions

- Keep the cover unnumbered and uncounted.
- Pre-textual pages are counted but their page numbers are hidden.
- Visible pagination starts at the Introduction and continues through the end.
- Figures: caption below.
- Tables: caption above.
- Every figure/table should be discussed in the text.
- The final Portuguese abstract should contain 150-500 words, in one paragraph, without citations.
- Update the cataloging sheet and committee information close to submission.
