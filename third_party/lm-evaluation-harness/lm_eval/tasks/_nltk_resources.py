"""Validate local IFEval resources without performing implicit downloads."""
from importlib.metadata import version

from packaging.version import Version


def require_punkt_tab():
    import nltk

    if Version(version("nltk")) < Version("3.9.1"):
        raise RuntimeError("IFEval requires nltk>=3.9.1; install the evaluator [ifeval] extra")
    try:
        # Checking English files catches an empty/partially copied directory.
        for name in ("collocations.tab", "sent_starters.txt", "abbrev_types.txt", "ortho_context.tab"):
            nltk.data.find("tokenizers/punkt_tab/english/" + name)
    except LookupError as error:
        raise LookupError(
            "IFEval needs local NLTK punkt_tab/english resources. Prepare them outside "
            "the GPU/offline task with: python -m nltk.downloader -d /path/to/nltk_data punkt_tab; "
            "then set NLTK_DATA=/path/to/nltk_data. Evaluation never downloads resources."
        ) from error
