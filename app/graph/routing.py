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