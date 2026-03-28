"""
Supervisor Multi-Agent ML Pipeline for Kaggle Competitions

This module implements a multi-agent architecture using langgraph-supervisor
where a supervisor agent coordinates specialized worker agents for different
ML pipeline tasks.

Architecture:
    Supervisor Agent (coordinates workflow)
        |
        +-- EDA Agent (data analysis)
        |       - tool_eda_analyze: Load and analyze train/test data
        |       - tool_eda_save_report: Save EDA summary
        |
        +-- Train Agent (model training)
        |       - tool_train_model: Train RandomForest classifier
        |
        +-- Eval Agent (model evaluation)
        |       - tool_eval_model: Calculate accuracy, F1, confusion matrix
        |       - tool_eval_save_metrics: Save metrics to JSON
        |
        +-- Submit Agent (submission creation)
                - tool_submit_create: Create submission.csv
                - tool_submit_validate: Validate submission format

Key Features:
    - Supervisor coordinates agents via handoff tools
    - Each agent has specialized tools for its domain
    - Session-based artifact storage with timestamps
    - Fallback pipeline if supervisor/LLM fails
    - Direct Kaggle submission (no agent)

Workflow:
    1. Supervisor calls EDA Agent -> analyze data
    2. Supervisor calls Train Agent -> train model
    3. Supervisor calls Eval Agent -> evaluate model
    4. Supervisor calls Submit Agent -> create submission
    5. Direct Kaggle submit and wait for results
    6. Generate final report

Usage:
    .venv/bin/python ai_agents_course/final_project/ai_agent_step_by_step/03_.py

Requirements:
    - OPENROUTER_API_KEY in .env
    - API_KAGGLE_KEY in .env (new token format: KGAT_xxx)
    - langgraph-supervisor package

Environment Variables:
    - OPENROUTER_API_KEY: API key for OpenRouter
    - API_KAGGLE_KEY: Your Kaggle API token (new format: KGAT_xxx)
    - KAGGLE_COMPETITION: Competition name (default: mws-ai-agents-2026)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

model_llm = "z-ai/glm-4.7-flash"
model_embedding = "google/gemini-embedding-001"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

# Имена файлов данных
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUBMISSION_FILE = "sample_submition.csv"

# Kaggle competition configuration
# Can be overridden via environment variable KAGGLE_COMPETITION
COMPETITION = os.getenv("KAGGLE_COMPETITION", "mws-ai-agents-2026")

logger: logging.Logger | None = None

# Global state for tools to access
_state: dict[str, Any] = {}


# ============================================================================
# SETUP ФУНКЦИИ
# ============================================================================

def _setup_logging(session_dir: Path) -> None:
    """Настройка логирования: консоль + файл в сессии."""
    global logger
    import sys
    log_file = session_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging to %s", log_file)


def _create_session_dir() -> Path:
    """Создает папку сессии с timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = ARTIFACTS_DIR / "sessions" / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    # Подпапки
    (session_dir / "code").mkdir(exist_ok=True)
    (session_dir / "models").mkdir(exist_ok=True)
    (session_dir / "reports").mkdir(exist_ok=True)

    return session_dir


def _get_llm():
    """LLM из config (model_llm). Провайдер — OpenRouter через ChatOpenAI."""
    try:
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv

        # Загружаем .env из корня проекта
        env_path = SCRIPT_DIR / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env")

        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model_llm,
            temperature=0
        )
    except Exception as e:
        if logger:
            logger.warning("LLM unavailable (%s); using fallback functions.", e)
        return None


# ============================================================================
# TOOLS ДЛЯ EDA AGENT
# ============================================================================

def eda_load_and_analyze(train_path: str, test_path: str, session_dir: str) -> str:
    """Load train and test data and perform basic analysis.

    Args:
        train_path: Path to train.csv
        test_path: Path to test.csv
        session_dir: Path to session directory for saving reports

    Returns:
        Summary of data analysis
    """
    try:
        import pandas as pd

        train_path = Path(train_path)
        test_path = Path(test_path)
        session_dir = Path(session_dir)

        results = []

        # Load and analyze train data (20% subset)
        if train_path.exists():
            train_df_full = pd.read_csv(train_path)
            train_df = train_df_full.sample(frac=0.2, random_state=42)

            results.append(f"Train data: {len(train_df)} rows (20% subset from {len(train_df_full)} total)")
            results.append(f"Train columns: {list(train_df.columns)}")
            results.append(f"Train shape: {train_df.shape}")
            results.append(f"Train dtypes:\n{train_df.dtypes.to_string()}")
            results.append(f"Missing values:\n{train_df.isnull().sum().to_string()}")

            # Save to global state
            _state["train_df"] = train_df
            _state["train_shape"] = train_df.shape
            _state["train_columns"] = list(train_df.columns)
        else:
            results.append(f"ERROR: Train file not found: {train_path}")

        # Load and analyze test data
        if test_path.exists():
            test_df = pd.read_csv(test_path)
            results.append(f"Test data: {len(test_df)} rows")
            results.append(f"Test columns: {list(test_df.columns)}")
            results.append(f"Test shape: {test_df.shape}")

            _state["test_df"] = test_df
            _state["test_shape"] = test_df.shape
        else:
            results.append(f"ERROR: Test file not found: {test_path}")

        summary = "\n".join(results)
        _state["eda_summary"] = summary

        if logger:
            logger.info("EDA completed: train_shape=%s, test_shape=%s",
                       _state.get("train_shape"), _state.get("test_shape"))

        return summary

    except Exception as e:
        error_msg = f"ERROR in eda_load_and_analyze: {e}"
        if logger:
            logger.error(error_msg)
        return error_msg


def eda_save_report(summary: str, session_dir: str) -> str:
    """Save EDA summary to a file.

    Args:
        summary: EDA summary text
        session_dir: Path to session directory

    Returns:
        Success message or error
    """
    try:
        session_dir = Path(session_dir)
        reports_dir = session_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        report_path = reports_dir / "eda_summary.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(summary)

        _state["eda_report_path"] = str(report_path)

        if logger:
            logger.info("EDA report saved to %s", report_path)

        return f"EDA report saved to {report_path}"

    except Exception as e:
        error_msg = f"ERROR saving EDA report: {e}"
        if logger:
            logger.error(error_msg)
        return error_msg


# ============================================================================
# TOOLS ДЛЯ TRAIN AGENT
# ============================================================================

def train_model(train_path: str, session_dir: str) -> str:
    """Train a RandomForest classification model.

    Args:
        train_path: Path to train.csv
        session_dir: Path to session directory for saving model

    Returns:
        Training results summary
    """
    try:
        import pandas as pd
        import joblib
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from catboost import CatBoostClassifier

        train_path = Path(train_path)
        session_dir = Path(session_dir)

        if not train_path.exists():
            return f"ERROR: Train file not found: {train_path}"

        # Load data (20% subset)
        train_df_full = pd.read_csv(train_path)
        train_df = train_df_full.sample(frac=0.2, random_state=42)

        # Find target column
        target_candidates = ["target", "label", "y"]
        target_col = None
        for c in target_candidates:
            if c in train_df.columns:
                target_col = c
                break
        if target_col is None:
            target_col = train_df.columns[-1]

        if logger:
            logger.info("Using target column: %s", target_col)

        # Prepare features (numeric only)
        X = train_df.drop(columns=[target_col], errors="ignore").select_dtypes(include=["number"])
        y = train_df[target_col]

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = CatBoostClassifier()
        model.fit(X_train, y_train)

        # Save model
        models_dir = session_dir / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Calculate training accuracy
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        # Save to state
        _state["model"] = model
        _state["model_path"] = str(model_path)
        _state["target_column"] = target_col
        _state["X_train_shape"] = X_train.shape
        _state["X_val_shape"] = X_val.shape
        _state["train_accuracy"] = train_acc
        _state["val_accuracy"] = val_acc

        result = (f"Model trained successfully!\n"
                 f"Target column: {target_col}\n"
                 f"Train shape: {X_train.shape}\n"
                 f"Val shape: {X_val.shape}\n"
                 f"Train accuracy: {train_acc:.4f}\n"
                 f"Val accuracy: {val_acc:.4f}\n"
                 f"Model saved to: {model_path}")

        if logger:
            logger.info("Model trained: train_acc=%.4f, val_acc=%.4f", train_acc, val_acc)

        return result

    except Exception as e:
        error_msg = f"ERROR in train_model: {e}"
        if logger:
            logger.error(error_msg)
        return error_msg


def train_get_feature_info() -> str:
    """Get feature names used by the trained model.

    Returns:
        Feature names or error
    """
    try:
        model = _state.get("model")
        if model is None:
            return "ERROR: No model found. Train model first."

        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
            _state["feature_names"] = features
            return f"Features used by model: {features}"
        else:
            return "Model does not have feature_names_in_ attribute"

    except Exception as e:
        return f"ERROR getting feature info: {e}"


# ============================================================================
# TOOLS ДЛЯ EVAL AGENT
# ============================================================================

def eval_model(model_path: str, train_path: str) -> str:
    """Evaluate the trained model on validation data.

    Args:
        model_path: Path to saved model
        train_path: Path to train.csv (to recreate split)

    Returns:
        Evaluation metrics
    """
    try:
        import pandas as pd
        import joblib
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

        model_path = Path(model_path)
        train_path = Path(train_path)

        if not model_path.exists():
            return f"ERROR: Model file not found: {model_path}"

        # Load model
        model = joblib.load(model_path)

        # Load data and recreate split
        train_df_full = pd.read_csv(train_path)
        train_df = train_df_full.sample(frandom_state=42) # берем весь датасет

        target_col = _state.get("target_column", train_df.columns[-1])
        X = train_df.drop(columns=[target_col], errors="ignore").select_dtypes(include=["number"])
        y = train_df[target_col]

        _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Make predictions
        y_pred = model.predict(X_val)

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        try:
            f1_macro = f1_score(y_val, y_pred, average="macro")
        except Exception:
            f1_macro = 0.0
        cm = confusion_matrix(y_val, y_pred)

        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "confusion_matrix": cm.tolist()
        }

        _state["local_metrics"] = metrics

        result = (f"Evaluation Results:\n"
                 f"Accuracy: {accuracy:.4f}\n"
                 f"F1-Macro: {f1_macro:.4f}\n"
                 f"Confusion Matrix:\n{cm}")

        if logger:
            logger.info("Evaluation: accuracy=%.4f, f1_macro=%.4f", accuracy, f1_macro)

        return result

    except Exception as e:
        error_msg = f"ERROR in eval_model: {e}"
        if logger:
            logger.error(error_msg)
        return error_msg

def eval_save_metrics(session_dir: str) -> str:
    """Save evaluation metrics to JSON file.

    Args:
        session_dir: Path to session directory

    Returns:
        Success message or error
    """
    try:
        session_dir = Path(session_dir)
        reports_dir = session_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        metrics = _state.get("local_metrics", {})
        if not metrics:
            return "ERROR: No metrics found. Run eval_model first."

        metrics_path = reports_dir / "local_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        _state["metrics_path"] = str(metrics_path)

        if logger:
            logger.info("Metrics saved to %s", metrics_path)

        return f"Metrics saved to {metrics_path}"

    except Exception as e:
        error_msg = f"ERROR saving metrics: {e}"
        if logger:
            logger.error(error_msg)
        return error_msg


# ============================================================================
# TOOLS ДЛЯ SUBMIT AGENT
# ============================================================================

def submit_create_submission(model_path: str, test_path: str, sample_submission_path: str, session_dir: str) -> str:
    """Create submission file for Kaggle.

    Args:
        model_path: Path to saved model
        test_path: Path to test.csv
        sample_submission_path: Path to sample_submission.csv
        session_dir: Path to session directory

    Returns:
        Submission file path or error
    """
    try:
        import pandas as pd
        import joblib
        import numpy as np

        model_path = Path(model_path)
        test_path = Path(test_path)
        sample_submission_path = Path(sample_submission_path)
        session_dir = Path(session_dir)

        if not model_path.exists():
            return f"ERROR: Model file not found: {model_path}"

        # Load model
        model = joblib.load(model_path)

        # Load test data
        test_df = pd.read_csv(test_path)

        # Prepare features (same as training)
        if hasattr(model, "feature_names_in_"):
            feature_cols = [c for c in model.feature_names_in_ if c in test_df.columns]
            X_test = test_df[feature_cols] if feature_cols else test_df.select_dtypes(include=[np.number])
        else:
            X_test = test_df.select_dtypes(include=[np.number])

        # Make predictions
        predictions = model.predict(X_test)

        # Create submission matching sample format
        if sample_submission_path.exists():
            sample = pd.read_csv(sample_submission_path)
            submission = sample.copy()
            pred_col = sample.columns[1] if len(sample.columns) > 1 else sample.columns[0]
            submission[pred_col] = predictions
            submission = submission[sample.columns]  # Keep only sample columns
        else:
            id_col = "id" if "id" in test_df.columns else test_df.columns[0]
            submission = test_df[[id_col]].copy() if id_col in test_df.columns else pd.DataFrame({"index": range(len(predictions))})
            submission["prediction"] = predictions

        # Save submission
        submission_path = session_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)

        _state["submission_path"] = str(submission_path)
        _state["submission_df"] = submission

        result = (f"Submission created successfully!\n"
                 f"Shape: {submission.shape}\n"
                 f"Columns: {list(submission.columns)}\n"
                 f"Saved to: {submission_path}")

        if logger:
            logger.info("Submission created: %s, shape=%s", submission_path, submission.shape)

        return result

    except Exception as e:
        error_msg = f"ERROR creating submission: {e}"
        if logger:
            logger.error(error_msg)
        return error_msg


def submit_validate(submission_path: str, sample_submission_path: str) -> str:
    """Validate submission format against sample.

    Args:
        submission_path: Path to submission.csv
        sample_submission_path: Path to sample_submission.csv

    Returns:
        Validation result
    """
    try:
        import pandas as pd

        submission_path = Path(submission_path)
        sample_submission_path = Path(sample_submission_path)

        if not submission_path.exists():
            return f"ERROR: Submission file not found: {submission_path}"

        if not sample_submission_path.exists():
            return f"ERROR: Sample submission file not found: {sample_submission_path}"

        submission = pd.read_csv(submission_path)
        sample = pd.read_csv(sample_submission_path)

        # Check columns match
        if list(submission.columns) != list(sample.columns):
            return (f"ERROR: Column mismatch!\n"
                   f"Submission columns: {list(submission.columns)}\n"
                   f"Expected columns: {list(sample.columns)}")

        # Check row count
        if len(submission) != len(sample):
            return (f"ERROR: Row count mismatch!\n"
                   f"Submission rows: {len(submission)}\n"
                   f"Expected rows: {len(sample)}")

        return f"Validation passed! Columns: {list(submission.columns)}, Rows: {len(submission)}"

    except Exception as e:
        return f"ERROR validating submission: {e}"


# ============================================================================
# LANGCHAIN TOOLS WRAPPERS
# ============================================================================

from langchain_core.tools import tool


@tool
def tool_eda_analyze(train_path: str, test_path: str, session_dir: str) -> str:
    """Load and analyze train and test data. Performs EDA and returns summary.

    Args:
        train_path: Path to train.csv file
        test_path: Path to test.csv file
        session_dir: Path to session directory for saving reports

    Returns summary of data analysis including shapes, columns, missing values.
    """
    return eda_load_and_analyze(train_path, test_path, session_dir)


@tool
def tool_eda_save_report(summary: str, session_dir: str) -> str:
    """Save EDA summary to a file in session_dir/reports/eda_summary.txt.

    Args:
        summary: EDA summary text to save
        session_dir: Path to session directory

    Returns success message with file path.
    """
    return eda_save_report(summary, session_dir)


@tool
def tool_train_model(train_path: str, session_dir: str) -> str:
    """Train a RandomForest classification model on the data.

    Uses 20% subset of training data. Splits 80/20 for train/validation.
    Saves model to session_dir/models/model.joblib.

    Args:
        train_path: Path to train.csv file
        session_dir: Path to session directory for saving model

    Returns training results including accuracy metrics.
    """
    return train_model(train_path, session_dir)


@tool
def tool_eval_model(model_path: str, train_path: str) -> str:
    """Evaluate the trained model on validation data.

    Calculates accuracy, F1-macro score, and confusion matrix.

    Args:
        model_path: Path to saved model.joblib file
        train_path: Path to train.csv to recreate train/val split

    Returns evaluation metrics.
    """
    return eval_model(model_path, train_path)


@tool
def tool_eval_save_metrics(session_dir: str) -> str:
    """Save evaluation metrics to session_dir/reports/local_metrics.json.

    Args:
        session_dir: Path to session directory

    Returns success message with file path.
    """
    return eval_save_metrics(session_dir)


@tool
def tool_submit_create(model_path: str, test_path: str, sample_path: str, session_dir: str) -> str:
    """Create submission file for Kaggle competition.

    Loads model, makes predictions on test data, creates CSV matching sample format.

    Args:
        model_path: Path to saved model.joblib file
        test_path: Path to test.csv file
        sample_path: Path to sample_submission.csv for format reference
        session_dir: Path to session directory for saving submission

    Returns submission file path and summary.
    """
    return submit_create_submission(model_path, test_path, sample_path, session_dir)


@tool
def tool_submit_validate(submission_path: str, sample_path: str) -> str:
    """Validate submission file format against sample submission.

    Checks that columns and row counts match.

    Args:
        submission_path: Path to created submission.csv
        sample_path: Path to sample_submission.csv

    Returns validation result (pass or error details).
    """
    return submit_validate(submission_path, sample_path)


# ============================================================================
# CREATE AGENTS
# ============================================================================

def create_agents(llm):
    """Create worker agents for ML pipeline."""
    from langgraph.prebuilt import create_react_agent

    # EDA Agent
    eda_agent = create_react_agent(
        llm,
        tools=[tool_eda_analyze, tool_eda_save_report],
        name="eda_agent",
        prompt=(
            "Ты ML-аналитик. Твоя задача — выполнить EDA (Exploratory Data Analysis). "
            "1. Используй tool_eda_analyze для загрузки и анализа данных. "
            "2. Используй tool_eda_save_report для сохранения отчета. "
            "Всегда сначала анализируй данные, потом сохраняй отчет. "
            "Возвращай краткое summary результатов."
        ),
    )

    # Train Agent
    train_agent = create_react_agent(
        llm,
        tools=[tool_train_model],
        name="train_agent",
        prompt=(
            "Ты ML-инженер. Твоя задача — обучить модель классификации. "
            "Используй tool_train_model для обучения RandomForest на 20% данных. "
            "Возвращай информацию о точности модели и пути к сохраненной модели."
        ),
    )

    # Eval Agent
    eval_agent = create_react_agent(
        llm,
        tools=[tool_eval_model, tool_eval_save_metrics],
        name="eval_agent",
        prompt=(
            "Ты ML-инженер. Твоя задача — оценить качество модели. "
            "1. Используй tool_eval_model для вычисления метрик (accuracy, F1, confusion matrix). "
            "2. Используй tool_eval_save_metrics для сохранения метрик в JSON. "
            "Возвращай основные метрики оценки."
        ),
    )

    # Submit Agent
    submit_agent = create_react_agent(
        llm,
        tools=[tool_submit_create, tool_submit_validate],
        name="submit_agent",
        prompt=(
            "Ты ML-инженер. Твоя задача — создать файл submission для Kaggle. "
            "1. Используй tool_submit_create для создания submission.csv. "
            "2. Используй tool_submit_validate для проверки формата. "
            "ВАЖНО: submission должен иметь те же колонки, что и sample_submission! "
            "Возвращай путь к созданному файлу."
        ),
    )

    return eda_agent, train_agent, eval_agent, submit_agent


# ============================================================================
# SUPERVISOR
# ============================================================================

SUPERVISOR_PROMPT = """Ты ML-техлид. Координируешь ML пайплайн для Kaggle соревнования.

Доступные специалисты:
- eda_agent: анализ данных (EDA), загрузка данных, статистика
- train_agent: обучение модели RandomForest, сохранение в joblib
- eval_agent: оценка модели (accuracy, F1-macro, confusion matrix)
- submit_agent: создание submission.csv в правильном формате

Доступные инструменты handoff:
- transfer_to_eda_agent(task) — передать задачу аналитику
- transfer_to_train_agent(task) — передать задачу на обучение
- transfer_to_eval_agent(task) — передать задачу на оценку
- transfer_to_submit_agent(task) — передать задачу на создание submission

Workflow (выполняй СТРОГО последовательно):
1. Вызови transfer_to_eda_agent для анализа train.csv и test.csv. Передай пути к файлам.
2. Дождись завершения EDA. Затем вызови transfer_to_train_agent для обучения модели.
3. Дождись завершения обучения. Затем вызови transfer_to_eval_agent для оценки.
4. Дождись завершения оценки. Затем вызови transfer_to_submit_agent для создания submission.
5. Когда submission создан — заверши работу.

ПРАВИЛА:
- За один ответ вызывай только ОДИН handoff инструмент
- Жди ответа от каждого агента перед переходом к следующему
- Передавай пути к файлам в task описании
- Если агент сообщает об ошибке — попробуй еще раз или переходи к следующему шагу

Пример task для eda_agent:
"Проанализируй данные: train_path=/path/to/train.csv, test_path=/path/to/test.csv, session_dir=/path/to/session"

Пример task для train_agent:
"Обучи модель: train_path=/path/to/train.csv, session_dir=/path/to/session"

Пример task для eval_agent:
"Оцени модель: model_path=/path/to/model.joblib, train_path=/path/to/train.csv, session_dir=/path/to/session"

Пример task для submit_agent:
"Создай submission: model_path=/path/to/model.joblib, test_path=/path/to/test.csv, sample_path=/path/to/sample_submition.csv, session_dir=/path/to/session"
"""


def create_supervisor_workflow(llm, agents):
    """Create supervisor workflow."""
    from langgraph_supervisor import create_supervisor

    eda_agent, train_agent, eval_agent, submit_agent = agents

    workflow = create_supervisor(
        [eda_agent, train_agent, eval_agent, submit_agent],
        model=llm,
        prompt=SUPERVISOR_PROMPT,
        include_agent_name="inline",
    )

    return workflow


# ============================================================================
# KAGGLE STEPS (БЕЗ AGENT)
# ============================================================================

def _load_kaggle_env() -> None:
    """
    Load Kaggle credentials from project root .env file.

    The .env file should contain:
    - API_KAGGLE_KEY: Your Kaggle API token (new format: KGAT_xxx)

    For new KGAT_ tokens, we use KAGGLE_API_TOKEN (not KAGGLE_KEY).
    KAGGLE_USERNAME is not required with the new token format.
    """
    from dotenv import load_dotenv

    # Load from project root .env
    # Path: ai_agents_course/final_project/ai_agent_step_by_step -> parent.parent.parent = project root
    project_root = SCRIPT_DIR
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        if logger:
            logger.info("Loaded environment from %s", env_path)
    else:
        if logger:
            logger.warning("No .env file found at %s", env_path)

    # Map API_KAGGLE_KEY to KAGGLE_API_TOKEN (required for KGAT_ tokens)
    # The new token format KGAT_xxx is a single token that replaces username+key
    api_kaggle_key = os.getenv("API_KAGGLE_KEY")
    if api_kaggle_key:
        if api_kaggle_key.startswith("KGAT_"):
            # New token format: use KAGGLE_API_TOKEN
            os.environ["KAGGLE_API_TOKEN"] = api_kaggle_key
            if logger:
                logger.info("Kaggle API token configured (KGAT_ format)")
        else:
            # Legacy token format: use KAGGLE_KEY + KAGGLE_USERNAME
            os.environ["KAGGLE_KEY"] = api_kaggle_key
            if logger:
                logger.info("Kaggle API key configured (legacy format)")
    else:
        if logger:
            logger.warning("API_KAGGLE_KEY not found in environment")


def kaggle_submit(state: dict) -> dict:
    """Отправка submission на Kaggle."""
    state = dict(state)
    sub_path = state.get("submission_path", "")
    if not sub_path or not Path(sub_path).exists():
        state["submit_ok"] = False
        state["submit_error"] = "No submission file"
        if logger:
            logger.warning("Kaggle submit: no submission file to submit.")
        return state

    _load_kaggle_env()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        state["submit_ok"] = False
        state["submit_error"] = "kaggle package not installed"
        if logger:
            logger.warning("Kaggle submit: kaggle not installed: %s", e)
        return state
    except Exception as e:
        state["submit_ok"] = False
        state["submit_error"] = f"Kaggle init/auth: {e}"
        if logger:
            logger.warning("Kaggle submit: Kaggle error: %s", e)
        return state

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        state["submit_ok"] = False
        state["submit_error"] = str(e)
        if logger:
            logger.error("Kaggle authenticate failed: %s", e)
        return state

    try:
        api.competition_submit(
            sub_path,
            "Submission from 03_.py supervisor pipeline",
            COMPETITION,
            quiet=False,
        )
        state["submit_ok"] = True
        state["submit_error"] = None
        if logger:
            logger.info("Kaggle submit: submission submitted to %s", COMPETITION)
    except Exception as e:
        state["submit_ok"] = False
        state["submit_error"] = str(e)
        if logger:
            logger.error("Kaggle submit failed: %s", e)
    return state


def kaggle_wait_results(state: dict) -> dict:
    """Подтверждение отправки submission.

    Примечание: Мы не получаем метрики с leaderboard - только подтверждаем,
    что файл был успешно отправлен в Kaggle.
    """
    state = dict(state)
    if not state.get("submit_ok"):
        state["public_score"] = None
        state["private_score"] = None
        state["submission_status"] = "not_submitted"
        return state

    if logger:
        logger.info("Kaggle wait: submission sent successfully!")
        logger.info("Check https://www.kaggle.com/competitions/%s for your score", COMPETITION)

    state["public_score"] = None
    state["private_score"] = None
    state["submission_status"] = "submitted_check_website"
    return state


# ============================================================================
# COLLECT RESULTS FROM FILES
# ============================================================================

def collect_results_from_files(session_dir: Path) -> dict:
    """Collect results from files created by agents.

    This is the reliable way to get results since global _state
    doesn't persist when tools run through langgraph agents.
    """
    results = {
        "eda_summary": "",
        "model_path": "",
        "local_metrics": {},
        "submission_path": "",
    }

    # Check EDA report
    eda_report_path = session_dir / "reports" / "eda_summary.txt"
    if eda_report_path.exists():
        try:
            results["eda_summary"] = eda_report_path.read_text(encoding="utf-8")
            if logger:
                logger.info("Collected EDA report from %s", eda_report_path)
        except Exception as e:
            if logger:
                logger.warning("Failed to read EDA report: %s", e)

    # Check model
    model_path = session_dir / "models" / "model.joblib"
    if model_path.exists():
        results["model_path"] = str(model_path)
        if logger:
            logger.info("Collected model path: %s", model_path)

    # Check metrics
    metrics_path = session_dir / "reports" / "local_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                results["local_metrics"] = json.load(f)
            if logger:
                logger.info("Collected metrics from %s", metrics_path)
        except Exception as e:
            if logger:
                logger.warning("Failed to read metrics: %s", e)

    # Check submission
    submission_path = session_dir / "submission.csv"
    if submission_path.exists():
        results["submission_path"] = str(submission_path)
        if logger:
            logger.info("Collected submission path: %s", submission_path)

    return results


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(state: dict) -> dict:
    """Generate final report."""
    state = dict(state)
    session_dir = Path(state.get("session_dir", "."))

    report = {
        "eda_summary": state.get("eda_summary", "")[:1000],
        "local_metrics": state.get("local_metrics", {}),
        "model_path": state.get("model_path", ""),
        "submission_path": state.get("submission_path", ""),
        "submit_ok": state.get("submit_ok"),
        "public_score": state.get("public_score"),
        "private_score": state.get("private_score"),
        "submission_status": state.get("submission_status", ""),
    }

    reports_dir = session_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    report_path_json = reports_dir / "final_report.json"
    report_path_txt = reports_dir / "final_report.txt"

    with open(report_path_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "Report 03 — Supervisor ML Pipeline",
        "===================================",
        "EDA summary: " + (report["eda_summary"] or "")[:500],
        "Local metrics: " + json.dumps(report["local_metrics"]),
        "Model: " + str(report["model_path"]),
        "Submission: " + str(report["submission_path"]),
        "Submitted: " + str(report["submit_ok"]),
        "Public score: " + str(report["public_score"]),
        "Private score: " + str(report["private_score"]),
        "Status: " + str(report["submission_status"]),
    ]
    text_report = "\n".join(lines)
    with open(report_path_txt, "w", encoding="utf-8") as f:
        f.write(text_report)

    state["report_path"] = str(report_path_txt)

    if logger:
        logger.info("Report saved to %s", report_path_txt)
    return state


# ============================================================================
# FALLBACK PIPELINE (БЕЗ LLM/SUPERVISOR)
# ============================================================================

def run_fallback_pipeline(state: dict) -> dict:
    """Run pipeline without supervisor (direct function calls)."""
    if logger:
        logger.info("Running fallback pipeline (no LLM/supervisor)...")

    session_dir = Path(state["session_dir"])
    train_path = state["train_path"]
    test_path = state["test_path"]
    sample_path = state["sample_submission_path"]

    # Step 1: EDA
    if logger:
        logger.info("Step 1: EDA (fallback)")
    eda_summary = eda_load_and_analyze(train_path, test_path, str(session_dir))
    eda_save_report(eda_summary, str(session_dir))

    # Step 2: Train
    if logger:
        logger.info("Step 2: Train (fallback)")
    train_result = train_model(train_path, str(session_dir))
    if logger:
        logger.info("Train result: %s", train_result[:200])

    # Step 3: Eval
    model_path = session_dir / "models" / "model.joblib"
    if model_path.exists():
        if logger:
            logger.info("Step 3: Eval (fallback)")
        eval_result = eval_model(str(model_path), train_path)
        eval_save_metrics(str(session_dir))
        if logger:
            logger.info("Eval result: %s", eval_result[:200])

    # Step 4: Submit
    if model_path.exists():
        if logger:
            logger.info("Step 4: Submission (fallback)")
        submit_result = submit_create_submission(str(model_path), test_path, sample_path, str(session_dir))
        if logger:
            logger.info("Submit result: %s", submit_result[:200])

    # Collect results from files
    file_results = collect_results_from_files(session_dir)
    state.update(file_results)

    return state


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline() -> dict[str, Any]:
    """Запуск supervisor ML pipeline."""
    from langgraph_supervisor import create_supervisor

    # Создаем сессию
    session_dir = _create_session_dir()

    # Настройка логирования
    _setup_logging(session_dir)
    if logger:
        logger.info("Session directory: %s", session_dir)

    # Инициализируем state
    state = {
        "session_dir": str(session_dir),
        "train_path": str(DATA_DIR / TRAIN_FILE),
        "test_path": str(DATA_DIR / TEST_FILE),
        "sample_submission_path": str(DATA_DIR / SAMPLE_SUBMISSION_FILE),
    }

    # Initialize global state
    _state.clear()
    _state.update(state)

    # Get LLM
    llm = _get_llm()

    if llm:
        try:
            if logger:
                logger.info("=" * 60)
                logger.info("Creating agents and supervisor...")

            # Create agents
            agents = create_agents(llm)

            # Create supervisor
            workflow = create_supervisor_workflow(llm, agents)
            app = workflow.compile()

            # Prepare initial task
            initial_task = f"""Запусти ML пайплайн для Kaggle соревнования.
train_path={state['train_path']}
test_path={state['test_path']}
sample_submission_path={state['sample_submission_path']}
session_dir={state['session_dir']}

Начни с EDA анализа данных, затем обучи модель, оцени и создай submission."""

            if logger:
                logger.info("=" * 60)
                logger.info("Running supervisor pipeline...")
                logger.info("Initial task: %s", initial_task[:200])

            # Run supervisor
            # Note: recursion_limit reduced to prevent infinite loops
            # Supervisor workflow needs: EDA -> Train -> Eval -> Submit = ~8-12 steps
            config = {"recursion_limit": 15}
            result = app.invoke(
                {"messages": [{"role": "user", "content": initial_task}]},
                config=config
            )

            # Extract results from messages
            messages = result.get("messages", [])
            if logger:
                logger.info("=" * 60)
                logger.info("Supervisor completed. Total messages: %d", len(messages))
                for i, msg in enumerate(messages[-5:]):
                    msg_type = type(msg).__name__
                    content = getattr(msg, "content", str(msg))[:200]
                    logger.info("Message %d [%s]: %s", i, msg_type, content)

            # Collect results from files (reliable way)
            # Global _state doesn't persist when tools run through langgraph agents
            if logger:
                logger.info("Collecting results from files...")
            file_results = collect_results_from_files(session_dir)

            # Check if agents actually created files
            # If not, run fallback pipeline (agents may have hallucinated)
            if not file_results["model_path"] or not file_results["submission_path"]:
                if logger:
                    logger.warning("Agents did not create required files. Running fallback pipeline...")
                state = run_fallback_pipeline(state)
            else:
                state["model_path"] = file_results["model_path"]
                state["submission_path"] = file_results["submission_path"]
                state["local_metrics"] = file_results["local_metrics"]
                state["eda_summary"] = file_results["eda_summary"]

                if logger:
                    logger.info("Results collected: model=%s, submission=%s",
                               bool(state["model_path"]), bool(state["submission_path"]))

        except Exception as e:
            if logger:
                logger.error("Supervisor pipeline failed: %s", e)
            logger.info("Falling back to direct pipeline...")
            state = run_fallback_pipeline(state)
    else:
        if logger:
            logger.info("No LLM available, running fallback pipeline...")
        state = run_fallback_pipeline(state)

    # Kaggle submit (direct call)
    if logger:
        logger.info("=" * 60)
        logger.info("Submitting to Kaggle...")
    state = kaggle_submit(state)

    # Wait for results
    if logger:
        logger.info("=" * 60)
        logger.info("Waiting for Kaggle results...")
    state = kaggle_wait_results(state)

    # Generate report
    if logger:
        logger.info("=" * 60)
        logger.info("Generating final report...")
    state = generate_report(state)

    if logger:
        logger.info("=" * 60)
        logger.info("Pipeline finished. Session: %s", session_dir)
        logger.info("Report: %s", state.get("report_path", "N/A"))

    return state


if __name__ == "__main__":
    run_pipeline()
