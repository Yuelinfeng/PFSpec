# PFSpec: Overleaf source package

This package contains a complete English manuscript and separate supplementary material in the official CVPR 2027 review format. It is a manuscript draft for author review, not a submitted paper.

## Open in Overleaf

1. Choose **New Project → Upload Project** and upload the ZIP.
2. Set the main document to **main.tex** and the compiler to **pdfLaTeX**. Use a current TeX Live version.
3. Recompile. Overleaf runs BibTeX automatically for `references.bib`.
4. To produce the separate supplementary PDF, change the main document to **supplementary.tex** and recompile. Restore **main.tex** when editing the main paper.

## Contents

- `main.tex`: anonymous main paper, LLM Usage Statement, and bibliography.
- `supplementary.tex`: complete supplementary analysis, additional results, implementation details, and 3D outputs.
- `metadata.tex`: paper title and conference information. Authors are anonymous by default.
- `preamble.tex`: shared packages and mathematical notation.
- `sections/`: editable English text divided by section.
- `references.bib`: 18 academic references with publication metadata and primary-source links.
- `figures/`: all four figures required for compilation. The two explanatory diagrams also include editable SVG sources. The qualitative 3D images retain the original renderings.
- `cvpr.sty` and `ieeenat_fullname.bst`: unmodified files from the official CVPR 2027 author kit.

## Compiled previews

The main PDF has eight pages in total, including the LLM Usage Statement and references. Natural-EOS timing and completion reference-image statistics share Table 5, and references follow the conclusion without a forced page break. The separate supplementary PDF has eight pages. Both were compiled with the locally bundled Tectonic engine and reviewed as page images. Overleaf uses pdfLaTeX by default; its exact line breaks may differ slightly.

## Scientific scope

The English manuscript preserves the Chinese manuscript's experimental values and distinguishes same-backend decoding, execution attribution, capped complete requests, and earlier natural-EOS basic P/F experiments. It adds no measured results. The natural-EOS study excludes Position Rescue and uses an earlier backend. The current full-method natural-EOS increment remains unmeasured. Internal project paths and experimental logs are not manuscript citations.

The two roots intentionally produce separate documents. Supplementary sections A--C are referred to by name from the main paper, so each PDF compiles independently without cross-document auxiliary files. Local compilation artifacts and internal revision notes are excluded from this package.

## Before submission

The current source uses anonymous review mode. The author team should complete its scientific review, confirm the LLM Usage Statement describes the full extent of tool use in the research, and enter author information only for an appropriate non-anonymous version. The statement currently describes the manuscript-preparation assistance known in this session. No submission identifiers or author identities have been invented.

## Template source

Official author guidelines: https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines

Official author kit: https://media.eventhosts.cc/Conferences/CVPR2027/author-kit-CVPR2027-v1-latex.zip

The supplied conference style and bibliography style retain their original notices. No margins, body font sizes, or conference style definitions have been changed.
