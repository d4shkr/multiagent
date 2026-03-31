# Multi-Agent Kaggle Solver for Tabular Regression

Supervisor Multi-Agent ML Pipeline for Kaggle Competitions 


## Архитектура

В системе четыре роли.

`ExplorerAgent` анализирует структуру датасета, пропуски, leakage-риски и характеристики target.

`EngineerAgent` выбирает feature subset, собирает preprocessing, прогоняет несколько регрессионных model-tools через CV и сохраняет лучший pipeline.

`EvaluatorAgent` делает holdout-оценку, schema/drift проверки, robustness checks и либо подтверждает readiness, либо отдает feedback на новую итерацию.

`SupervisorAgent` замыкает цикл Explorer -> Engineer -> Evaluator и повторяет train/eval, пока качество не пройдет quality gates или не будет достигнут лимит итераций.

## Tool layer

Код доменной логики вынесен в tools, а агенты выбирают и комбинируют их. Основные tools:

- `profile_dataset`
- `detect_leakage`
- `summarize_missingness`
- `select_feature_subset`
- `build_preprocessor`
- `train_ridge`
- `train_random_forest`
- `train_gradient_boosting`
- `train_extra_trees`
- `run_regression_cv`
- `evaluate_holdout`
- `check_schema`
- `compute_drift`
- `generate_submission`

## Метрики

Основные метрики проекта:

- CV RMSE
- Holdout RMSE
- Holdout MAE
- Holdout $R^2$
- robustness shift
- mean train-test drift

Для выбора лучшей модели используется минимизация RMSE.

## Feedback loop

`EvaluatorAgent` формирует feedback, если:

- RMSE слишком высокий;
- $R^2$ слишком низкий;
- модель слишком чувствительна к малому шуму;
- test schema слишком сильно расходится с train schema.

Тогда `SupervisorAgent` запускает следующую итерацию `EngineerAgent`, который исключает часть неудачных кандидатов и пробует новый набор tools.

## Open-source LLM

По умолчанию используется open-source модель `qwen/qwen2.5-7b-instruct`. При наличии OpenRouter или Ollama планирование делается через OSS LLM. Если LLM недоступна, проект не падает и использует детерминированный planner.

## RAG

Retrieval выполняется по локальному индексу в `rag_storage/`. Ноутбуки взяты с Kaggle или написаны нами.

## Запуск

```bash
source setup.sh
```
```bash
source run.sh
```
