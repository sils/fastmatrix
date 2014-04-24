default: help

help:
	@echo "---------------------------------------------------------------"
	@echo "Documentary targets:"
	@echo " help        : show this help text"
	@echo "---------------------------------------------------------------"
	@echo "Dial helpers:"
	@echo " externdial  : dial TU server from the outside"
	@echo " tudial      : dial TU server from the TUHH"
	@echo "---------------------------------------------------------------"

externdial:
	@scripts/externdial.sh

tudial:
	@scripts/tudial.sh

