file="paper"

pdflatex $file && bibtex $file && pdflatex $file && bibtex $file && pdflatex $file

rm *.log *.aux *.bbl *.blg *.out *.fls *.fdb_latexmk