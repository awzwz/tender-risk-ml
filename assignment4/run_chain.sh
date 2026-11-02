#!/bin/zsh
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for nbn in P3_model_comparison_tuning P4_transfer_learning P5_fine_tuning P6_llm_lora; do
  echo "=== $nbn $(date +%T)"
  ~/.venvs/tender-risk-ml/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 $nbn.ipynb > logs/$nbn.log 2>&1
  echo "exit $? $(date +%T)"; tail -2 logs/$nbn.log
done
