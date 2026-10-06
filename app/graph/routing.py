def route_after_analysis(state):

    sources = state.get(
        "sources",
        []
    )

    if not sources:
        return "generate"

    if len(sources) == 1:
        return "retrieve"

    return "parallel_retrieve"

def route_after_document_grade(state):

    if state.get(
        "document_relevant",
        False
    ):
        return "generate"

    retry_count = state.get(
        "retry_count",
        0
    )

    if retry_count >= 3:
        return "fallback"

    return "rewrite"