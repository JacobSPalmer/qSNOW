.PHONY: uml

uml:
	mkdir -p uml
	pyreverse -o png -p qSNOW -d uml src/qsnow/interface/chip.py src/qsnow/interface/models.py
