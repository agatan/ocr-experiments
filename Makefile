ifdef CUDA_VERSION
	PLATFORM_SPECIFIED_REQUIREMENTS=requirements-gpu.txt
else
	PLATFORM_SPECIFIED_REQUIREMENTS=requirements-cpu.txt
endif

.PHONY: dep
dep: $(PLATFORM_SPECIFIED_REQUIREMENTS) requirements.txt
	pip install --user -r $(PLATFORM_SPECIFIED_REQUIREMENTS)
	pip install --user -r requirements.txt


.PHONY: dep-dev
dep-dev: requirements-dev.txt
	pip install --user -r requirements-dev.txt
