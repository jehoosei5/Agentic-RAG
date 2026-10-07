from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.types import Send

from app.graph.state import AgentState

from app.graph.nodes import (
    analyze_query,
    retrieve,
    retrieve_worker,
    grade_documents,
    rewrite_query,
    generate_answer,
    grade_answer,
    regenerate_answer,
    fallback,
)

from app.graph.routing import (
    route_after_analysis,
    route_after_document_grade,
    route_after_answer_grade,
)


def parallel_retrieve(state):

    question = state["question"]

    sources = state["sources"]

    return [
        Send(
            "retrieve_worker",
            {
                "question": question,
                "source": source,
            }
        )
        for source in sources
    ]


def route_analyzed_query(state):

    route = route_after_analysis(state)

    if route == "parallel_retrieve":
        return parallel_retrieve(state)

    return route


def build_graph():

    graph = StateGraph(
        AgentState
    )

    # -------------------------
    # Nodes
    # -------------------------

    graph.add_node(
        "analyze",
        analyze_query
    )

    graph.add_node(
        "retrieve",
        retrieve
    )

    graph.add_node(
        "retrieve_worker",
        retrieve_worker
    )

    graph.add_node(
        "grade_documents",
        grade_documents
    )

    graph.add_node(
        "rewrite",
        rewrite_query
    )

    graph.add_node(
        "generate",
        generate_answer
    )

    graph.add_node(
        "grade_answer",
        grade_answer
    )

    graph.add_node(
        "regenerate",
        regenerate_answer
    )

    graph.add_node(
        "fallback",
        fallback
    )

    # -------------------------
    # Start
    # -------------------------

    graph.add_edge(
        START,
        "analyze"
    )

    # -------------------------
    # Query routing
    # -------------------------

    graph.add_conditional_edges(
        "analyze",
        route_analyzed_query,
        {
            "generate": "generate",
            "retrieve": "retrieve",
        }
    )

    # -------------------------
    # Single retrieval
    # -------------------------

    graph.add_edge(
        "retrieve",
        "grade_documents"
    )

    # -------------------------
    # Parallel retrieval
    # -------------------------

    graph.add_edge(
        "retrieve_worker",
        "grade_documents"
    )

    # -------------------------
    # Document grading
    # -------------------------

    graph.add_conditional_edges(
        "grade_documents",
        route_after_document_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "fallback",
        }
    )

    # -------------------------
    # Query rewriting loop
    # -------------------------

    graph.add_edge(
        "rewrite",
        "retrieve"
    )

    # -------------------------
    # Answer generation
    # -------------------------

    graph.add_edge(
        "generate",
        "grade_answer"
    )

    # -------------------------
    # Answer grading
    # -------------------------

    graph.add_conditional_edges(
        "grade_answer",
        route_after_answer_grade,
        {
            "end": END,
            "regenerate": "regenerate",
        }
    )

    # -------------------------
    # Regeneration loop
    # -------------------------

    graph.add_edge(
        "regenerate",
        "grade_answer"
    )

    # -------------------------
    # Fallback
    # -------------------------

    graph.add_edge(
        "fallback",
        END
    )

    return graph.compile()