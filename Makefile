# ===== VARIABLES =====
# include the configuration file
include Makefile.config

# Python version file
PYTHON_VERSION_FILE = .python-version

# Python interpreters
PYENV_PYTHON = $(shell pyenv root)/versions/$(PYTHON_VERSION)/bin/python
PYTHON = $(VENV_DIR)/bin/python

# requirements
COMPILED_REQUIREMENTS_FILE = requirements.txt
ifeq ($(DEV), true)
	REQUIREMENTS_FILES = requirements.in requirements-dev.in
else
	REQUIREMENTS_FILES = requirements.in
endif

# virtual environment
VENV_DIR = venv
VENV_ACTIVATE = $(VENV_DIR)/bin/activate

# data directories
DATA_DIR = data
RAW_DATA_DIR = $(DATA_DIR)/raw

# raw data zip
RAW_DATA_ZIP = $(RAW_DATA_DIR)/$(COMPETITION_NAME).zip


# ===== INIT & CLEAR =====
.PHONY : init
init :
	$(MAKE) install-requirements
ifeq ($(LOAD_DATA_ON_INIT), true)
	$(MAKE) load-data
endif

.PHONY : clear
clear :
	rm -rf .python-version requirements.txt venv


## ===== JUPYTER =====
# start a Jupyter server
.PHONY : jupyter
jupyter : | $(VENV_ACTIVATE)
	. $(VENV_ACTIVATE) && ipython kernel install --user --name=$(VENV_DIR) && env PYTHONPATH=`pwd` jupyter notebook


# ===== DATA =====
# download the raw data, unzip it and delete the archive
.PHONY : load-data
load-data : unzip-raw-data

# remove data directory
.PHONY : clear-data
clear-data :
	rm -rf $(DATA_DIR)

# download the raw data zip using the Kaggle API
$(RAW_DATA_ZIP) : | $(RAW_DATA_DIR) $(VENV_ACTIVATE)
	. $(VENV_ACTIVATE) && kaggle competitions download $(COMPETITION_NAME) -p $(RAW_DATA_DIR)

# unzip the raw data archive and delete it
.PHONY : unzip-raw-data
unzip-raw-data : | $(RAW_DATA_ZIP)
	unzip $| -d $(RAW_DATA_DIR)
	rm $|

# create the raw data directory
$(RAW_DATA_DIR) :
	mkdir -p $@


# ===== PYTHON =====
# install requirements
.PHONY : install-requirements
install-requirements : $(COMPILED_REQUIREMENTS_FILE)
	. $(VENV_ACTIVATE) && pip-sync

# compile requirements
$(COMPILED_REQUIREMENTS_FILE) : $(REQUIREMENTS_FILES) | $(VENV_ACTIVATE)
	. $(VENV_ACTIVATE) && pip-compile --output-file=$@ --strip-extras $^
	
# create virtual environment
$(VENV_ACTIVATE) : $(PYTHON_VERSION_FILE)
	$(PYENV_PYTHON) -m pip install -U pip virtualenv
	$(PYENV_PYTHON) -m virtualenv $(VENV_DIR)
	$(PYTHON) -m pip install -U pip pip-tools

# set pyenv local Python version
$(PYTHON_VERSION_FILE) : | $(PYENV_PYTHON)
	pyenv local $(PYTHON_VERSION)

# install Python version
$(PYENV_PYTHON) :
	arch -arm64 pyenv install --skip-existing $(PYTHON_VERSION)
