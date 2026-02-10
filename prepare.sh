#!/bin/bash
set -euo pipefail

pip install transformers==4.46.3
pip install sentencepiece
pip install openai
pip install datasets
pip install openpyxl
pip install bitsandbytes==0.43.3
pip install 'accelerate>=0.26.0'
pip install seaborn
