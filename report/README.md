# LaTeX report

Compile (needs a TeX distribution with pdflatex, or upload this folder to Overleaf):

    python train.py                  # from the repo root, creates report/cv_metrics.csv
    python report/make_assets.py     # regenerates figures/ and the numbers used in the text
    cd report && pdflatex report.tex && pdflatex report.tex

Edit `\reportauthor` and `\repourl` near the top of `report.tex`.
All numbers in the text and tables come from your own run (generated.tex, CVMeanRows.tex, CVFoldRows.tex).
