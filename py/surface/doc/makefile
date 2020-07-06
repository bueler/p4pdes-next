all: blow.pdf

blow.pdf: blow.tex
	pdflatex blow.tex
	pdflatex blow.tex

.PHONY: clean

clean:
	@rm -f *~ *.aux *.log *.synctex.gz
