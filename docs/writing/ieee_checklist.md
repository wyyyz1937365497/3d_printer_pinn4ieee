# IEEE Format Checklist

**Purpose**: Complete checklist for ensuring IEEE conference/journal paper compliance.

---

## Document Structure

### Page Layout

- [ ] **Two-column format** for final submission
- [ ] **Single-column** for review (if required)
- [ ] **Page size**: US Letter (8.5" × 11")
- [ ] **Margins**:
  - Top/Bottom: 1"
  - Sides: 0.75"
  - Gutter: 0.25" (for double-sided)
- [ ] **Column width**: 3.25" (standard) or 3.5"
- [ ] **Column separation**: 0.35" or 0.33"
- [ ] **No column breaks** in middle of equations
- [ ] **No widows/orphans** (single lines at page breaks)

### Section Organization

- [ ] **Title page** with authors and affiliations
- [ ] **Abstract** (150-250 words for conferences)
- [ ] **Keywords** (4-6 keywords)
- [ ] **Sections numbered** (I, II, III or 1, 1.1, 1.1.1)
- [ ] **Introduction** (Section I)
- [ ] **Related Work** (Section II or part of Introduction)
- [ ] **Methodology** (Section III)
- [ ] **Experiments** (Section IV)
- [ ] **Results** (Section V)
- [ ] **Discussion** (Section VI or combined with Results)
- [ ] **Conclusion** (Section VII or last)
- [ ] **Acknowledgments** (optional, before References)
- [ ] **References** (numbered or alphabetical)

---

## Text Formatting

### Font Requirements

- [ ] **Font**: Times New Roman (or similar serif)
- [ ] **Font size**: 10 pt for body text
- [ ] **Title size**: 14-18 pt (depending on template)
- [ ] **Section headings**: Bold, 12 pt
- [ ] **Subsection headings**: Bold, 10 pt
- [ ] **Captions**: 9 pt
- [ ] **Table entries**: 9 pt
- [ ] **Footnotes**: 8 or 9 pt

### Line Spacing

- [ ] **Single spacing** for body text
- [ ] **No extra space** between paragraphs (use indentation)
- [ ] **Paragraph indentation**: 0.25" or 1 pica
- [ ] **6 pt spacing** before/after section headings (check template)

### Headings

- [ ] **Section headings**: Centered, Roman numerals (I, II, III)
  - Format: `I. INTRODUCTION`
  - Font: Bold, all caps
- [ ] **Subsection headings**: Left-aligned, Arabic numerals (1.1, 1.2)
  - Format: `A. System Overview` or `1.1 System Overview`
  - Font: Bold, title case
- [ ] **Sub-subsection headings**: Left-aligned, italic
  - Format: `1) Trajectory Dynamics` or `1.1.1) Dynamics`
  - Font: Italic, title case
- [ ] **Consistent hierarchy** throughout

---

## Figures

### Figure Placement

- [ ] **Figures numbered** sequentially (Fig. 1, Fig. 2, ...)
- [ ] **Figures referenced** in text BEFORE they appear
- [ ] **Captions below** figures
- [ ] **Centered** within column (or across two columns if `figure*`)
- [ ] **No orphan figures** (figures separated from text)
- [ ] **Top or bottom** of page preferred
- [ ] **Two-column figures** placed at top or bottom of page

### Figure Quality

- [ ] **Resolution**: At least 300 DPI for raster images
- [ ] **Vector format** preferred (PDF, EPS, SVG)
- [ ] **Grayscale** or color (ensure readability in both)
- [ ] **Font size**: Minimum 8 pt in final figure
- [ ] **Line weight**: Thick enough to be visible when scaled
- [ ] **Axis labels**: Clear, with units
- [ ] **Legends**: Readable, positioned appropriately
- [ ] **No color-dependence** for critical information

### Figure Captions

- [ ] **Format**: `Figure 1. Description of the figure.`
- [ ] **Concise but descriptive**
- [ ] **Bold** "Figure X." prefix
- [ ] **Period** at end
- [ ] **Self-contained** (understandable without main text)
- [ ] **Multi-line captions**: Hanging indent (if more than one line)

### Figure File Formats

Preferred (in order):
1. **PDF** (vector graphics, best quality)
2. **EPS** (vector graphics)
3. **PNG** (raster, with transparency support)
4. **JPG** (raster, avoid for line drawings)

Avoid:
- Word documents with embedded images
- Low-resolution screenshots
- Bitmap formats (BMP)

---

## Tables

### Table Placement

- [ ] **Tables numbered** sequentially (Table I, Table II, ...)
- [ ] **Tables referenced** in text BEFORE they appear
- [ ] **Captions above** tables
- [ ] **Centered** within column (or use `table*` for wide tables)
- [ ] **No orphan tables** (tables separated from text)

### Table Formatting

- [ ] **Use booktabs**: `\toprule`, `\midrule`, `\bottomrule`
- [ ] **No vertical lines** (IEEE style uses horizontal only)
- [ ] **Column headings**: Bold or italic
- [ ] **Alignment**: Left for text, right for numbers, centered for short entries
- [ ] **Decimal alignment** for numbers (using `dcolumn` package)
- [ ] **Consistent precision** in numerical data
- [ ] **Units** in column headings or table notes

### Table Captions

- [ ] **Format**: `Table I. Description of the table.`
- [ ] **Concise but descriptive**
- [ ] **Bold** "Table X." prefix
- [ ] **Period** at end
- [ ] **Placed above** the table

### Table Notes

- [ ] **Superscript letters** (a, b, c) for notes
- [ ] **Notes below** bottom line
- [ ] **Footnote symbols** (*, †, ‡) if superscripts insufficient
- [ ] **Explanatory notes** for abbreviations or special values

---

## Equations

### Equation Formatting

- [ ] **All equations numbered** sequentially (1), (2), (3)
- [ ] **Numbers in parentheses** (right-aligned)
- [ ] **Punctuation** after equations (comma or period)
- [ ] **Centered** on page
- [ ] **Variables**: Italic (e.g., $x$, $y$, $t$)
- [ ] **Functions/Operators**: Roman (e.g., $\sin$, $\cos$, $\max$, $\ln$)
- [ ] **Vectors**: Bold (e.g., $\mathbf{x}$, $\mathbf{v}$)
- [ ] **Matrices**: Bold uppercase (e.g., $\mathbf{A}$, $\mathbf{B}$)

### Equation References

- [ ] **Use \eqref{label}** for equation references
- [ ] **Parentheses** around equation number: "Eq. (5)"
- [ ] **Self-contained** references: "Equation (5) gives..."
- [ ] **No** "Eq. 5" or "equation 5" (use parentheses)

### Mathematical Notation

- [ ] **Consistent notation** throughout
- [ ] **Define all symbols** on first use
- [ ] **Use fractions** for simple expressions: $\frac{a}{b}$
- [ ] **Use inline fractions** for complex: $a/b$
- [ ] **Units** in text, not in equations: $x$ [mm], not $x \text{ mm}$
- [ ] **Subscripts**: Italic if variable ($m_x$), Roman if label ($T_{\text{nozzle}}$)
- [ ] **Superscripts**: Careful with exponents vs. footnote marks

---

## References

### Citation Style

- [ ] **Numbered references** [1], [2] or [1]-[3]
- [ ] **In-text citations**:
  - Single author: [1]
  - Two authors: [1], [2]
  - Three or more: [1]-[3]
  - Specific page: [1, p. 123]
- [ ] **Citation placement**:
  - Before punctuation: "as shown in [1]."
  - Multiple citations: [1], [3], [5]
  - Range of citations: [1]-[5]

### Reference Format (IEEE Style)

#### Journal Papers

```
[1] A. B. Author and C. D. Author, "Title of the paper," *Abbrev. Title of Journal*, vol. x, no. x, pp. xxx-xxx, Abbrev. Month, Year.
```

**Example**:
```
[1] A. Bell, B. Smith, and C. Johnson, "Trajectory errors in FDM 3D printers," *Int. J. Adv. Manuf. Technol.*, vol. 112, no. 3, pp. 1234-1245, Mar. 2024.
```

#### Conference Papers

```
[2] A. B. Author, "Title of paper," in *Proc. Abbrev. Conf. Title*, City, Country (or City, State), Year, pp. xxx-xxx.
```

**Example**:
```
[2] D. Lee, "Real-time error correction for 3D printing," in *Proc. IEEE Int. Conf. Robotics Autom. (ICRA)*, Singapore, 2023, pp. 1234-1239.
```

#### Books

```
[3] A. B. Author, *Title of Book*, xth ed. City of Publisher, Country: Publisher, Year.
```

#### Technical Reports

```
[4] A. B. Author, "Title of report," Abbrev. Tech. Rep., Abbrev. Lab., City of Lab., State, Country, Rep. XXX, Year.
```

#### Online Sources

```
[5] A. B. Author, "Title of article," *Title of Publication*, [Online]. Available: URL. [Accessed: Date].
```

### Reference List

- [ ] **In order of citation** (not alphabetical)
- [ ] **Hanging indent** for second and subsequent lines
- [ ] **All cited works included**
- [ ] **No uncited references**
- [ ] **DOIs included** when available
- [ ] **URLs working** (for online sources)
- [ ] **Author names**: Initial + last name (A. B. Author)
- [ ] **Paper titles**: In quotation marks
- [ ] **Publication names**: Italicized
- [ ] **Volume/issue numbers**: Italicized
- [ ] **Page ranges**: Inclusive (pp. 123-125, not 123-5)

---

## Common Mistakes to Avoid

### Formatting Issues

❌ **Don't**:
- Use color to convey information (won't work in print)
- Use low-resolution figures (< 300 DPI)
- Put captions below tables or above figures
- Use vertical lines in tables
- Number sections inconsistently
- Use excessive abbreviations without definition

✅ **Do**:
- Test grayscale printing of all figures
- Use vector graphics when possible
- Follow template exactly
- Be consistent with notation
- Define all abbreviations on first use
- Use spell checker

### Writing Issues

❌ **Don't**:
- Use first person excessively ("I", "we")
- Use contractions ("can't", "won't" → "cannot", "will not")
- Use vague quantifiers ("very", "quite", "rather")
- Overclaim results
- Ignore page limits
- Use informal language

✅ **Do**:
- Use active voice when appropriate
- Use precise, quantitative statements
- Acknowledge limitations honestly
- Stay within page limits
- Use formal, academic tone
- Have others proofread

---

## Pre-Submission Checklist

### Content Review

- [ ] **Title**: Concise, descriptive, < 15 words
- [ ] **Abstract**: 150-250 words, covers motivation, method, results
- [ ] **Keywords**: 4-6 relevant terms
- [ ] **Introduction**: Sets context, states problem, outlines contributions
- [ ] **Methods**: Sufficient detail for reproduction
- [ ] **Results**: Clear, quantified, with statistics
- [ ] **Discussion**: Interpretation, limitations, comparison
- [ ] **Conclusion**: Summary, implications, future work
- [ ] **References**: Complete, correct format, all cited

### Technical Review

- [ ] **All equations numbered** and referenced
- [ ] **All figures/tables numbered** and referenced
- [ ] **All abbreviations defined** on first use
- [ ] **Consistent notation** throughout
- [ ] **No orphan references** (citations without entries)
- [ ] **No broken equations** (display issues)
- [ ] **No typos** (use spell checker, proofread)

### Format Review

- [ ] **PDF created** from LaTeX (not Word if possible)
- [ ] **Font embedding** checked
- [ ] **Figure quality** verified (300+ DPI)
- [ ] **Table formatting** checked (no vertical lines)
- [ ] **Page limits** respected
- [ ] **Margins** correct (0.75" sides, 1" top/bottom)
- [ ] **Column format** correct (two-column)
- [ ] **No widows/orphans** in final PDF

### Final Checks

- [ ] **Read entire paper** from start to finish
- [ ] **Check for consistency** (notation, terminology, formatting)
- [ ] **Verify all URLs** in references
- [ ] **Check all DOIs** (if included)
- [ ] **Confirm author order** and affiliations
- [ ] **Proofread** (ask colleague if possible)
- [ ] **Check page numbers** in references (if article has them)

---

## IEEE Templates and Resources

### Official Resources

- **IEEE Conference Templates**: [IEEE Conference Template](https://www.ieee.org/conferences/publishing/templates)
- **IEEE Journal Templates**: [IEEE Article Templates](https://www.ieee.org/authors/publish/tools/te.html)
- **IEEE Reference Guide**: [IEEE Reference Guide PDF](https://ieeeauthorcenter.ieee.org/wp-content/uploads/IEEE-Reference-Guide.pdf)
- **LaTeX Template**: [IEEEtran LaTeX Template](https://www.ieee.org/conferences/publishing/ieee_authorn_guide_conf.pdf)

### Useful Packages

```latex
\usepackage{cite}        % For citations [1]-[3]
\usepackage{graphicx}    % For figures
\usepackage{booktabs}    % For professional tables
\usepackage{amsmath}     % For advanced math
\usepackage{siunitx}     % For units
\usepackage{algorithm}   % For algorithms
\usepackage{algpseudocode} % For algorithm pseudocode
```

### Common Commands

```latex
% Two-column figure
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.8\textwidth]{figure.pdf}
  \caption{Caption here.}
  \label{fig:wide}
\end{figure*}

% Algorithm
\begin{algorithm}
\caption{Algorithm Name}
\label{alg:name}
\begin{algorithmic}[1]
\State Initialize...
\If{condition}
    \State Do something...
\EndIf
\end{algorithmic}
\end{algorithm}

% Table with booktabs
\begin{table}[t]
\centering
\caption{Table Caption}
\begin{tabular}{lcr}
\toprule
Header 1 & Header 2 & Header 3 \\
\midrule
Data 1 & Data 2 & Data 3 \\
Data 4 & Data 5 & Data 6 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Last Minute Checks

### Before Final Submission

- [ ] **Remove all comments** from LaTeX source
- [ ] **Remove all TODO items** from text
- [ ] **Check for "et al."** vs. list all authors
- [ ] **Verify word count** (if required)
- [ ] **Check file size** (if uploading to system)
- [ ] **Test PDF opens** correctly on different computers
- [ ] **Email PDF to yourself** and verify

### At Conference/Journal System

- [ ] **Choose correct submission type**
- [ ] **Upload PDF** (or source files if required)
- [ ] **Upload figures separately** (if required)
- [ ] **Enter metadata** (title, authors, abstract)
- [ ] **Classify paper** (topic area)
- [ ] **Check for conflicts of interest**
- [ ] **Confirm copyright agreement**

---

## Document Status

**Checklist Version**: 1.0
**Last Updated**: 2026-02-02
**Target**: IEEE Conference/Journal Paper

**Completed**: ☐ All items checked
**Reviewer**: ___________________
**Date**: ___________________

---

## Related Documentation

- [Structure Template](../structure_template.md) - Overall paper structure
- [Section Templates](./) - Individual section templates
- [Formula Library](../../theory/formulas.md) - Equation reference
- [Bibliography](./bibliography.bib) - Reference management
