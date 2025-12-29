from functools import partial
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda

from summarization.workflow.graph.summary_base_models import SummaryState
from summarization.workflow.utils.langfuse_node_wrapper import wrap_node # New for wrapping nodes
# Import Main Nodes
from summarization.workflow.nodes.create_folder_dao_node import create_folder_dao
from summarization.workflow.nodes.enumerate_chapters_to_run_node import enumerate_chapters_to_run_node
from summarization.workflow.nodes.chapter_timeline_nodes.chapter_summary_node import ChapterTimelineNode, ChapterTimelineState
from summarization.workflow.nodes.chapter_timeline_nodes.chapter_coalesce_node import (
    chapter_summary_completed_node,
    chapter_summary_deep_merge_node
)
from summarization.workflow.nodes.pydantic_to_dict_node import pydantic_to_dict_node

# Import Edges
from summarization.workflow.edges.chapter_summary_routing import chapter_summary_routing

# Import Base Models
from summarization.workflow.graph.summary_base_models import (
    ChapterTimeline,
    ChapterTimelineConfig
)

def get_subgraph(ch):
    """Wrapper function to merge the summary state with the chapter timeline state

    Args:
        ch (str): The current chapter

    Returns:
        func: The function that will be invoked when the graph runs
    """
    subgraph = ChapterTimelineNode().build(ch)
    async def invoke_subgraph(summary_graph_state: SummaryState,
                                ch = ch) -> dict:
        """Invokes the chapter timeline subgraph

        Args:
            summary_graph_state (SummaryState): The state of the summary graph
            ch (str, optional): The chapter to run. Defaults to ch.

        Returns:
            dict: The subgraph state for the chapter timeline
        """
        chapter_timeline = ChapterTimeline(
            chapter = ch,
            chapter_timeline_config = ChapterTimelineConfig(
                **dict(summary_graph_state.config.timeline_config),
            )
        )

        chapter_timeline_state = ChapterTimelineState(
                    **dict(chapter_timeline),
                    folder_dao = summary_graph_state.folder_dao,
                    folder_num = summary_graph_state.folder_num,
                    internal_pmi_facr_available = summary_graph_state.internal_pmi_facr_available,
                    internal_all_pmi = [pmi for pmi,chap in summary_graph_state.internal_dig_to_chapter_map.items() if chap==ch],
        )

        s = await subgraph.ainvoke(chapter_timeline_state)

        return {'internal_list_of_chapter_timelines': [{ch: s}]}
    return invoke_subgraph

def build_graph(summary_config):
    """Build the graph for summarization

    Args:
        summary_config (SummaryConfig): The config for the summarization

    Returns:
        Graph: The graph for summarization of the chapter timeline
    """
    chapter_summary_nodes = {}

    for ch in summary_config.timeline_config.available_chapters:

        invoke_subgraph = get_subgraph(ch)
        # wrap the raw function before turning it into a RunnableLambda
        wrapped_invoke = wrap_node(invoke_subgraph, f'chapter_{ch}_summary_node')
        subgraph_lambda = RunnableLambda(wrapped_invoke)
        chapter_summary_nodes[f'chapter_{ch}_summary_node'] = subgraph_lambda

    # Build Initial State
    builder = StateGraph(SummaryState)

    # Add Initial edge from START to first node
    builder.add_edge(START, 'create_folder_dao')

    # # Add Sequence of Nodes
    # builder.add_sequence([
    #     # Create Folder DAO
    #     ('create_folder_dao', create_folder_dao),
    #     # Find chapters with PMIs that we want to summarize
    #     ('enumerate_chapters_to_run_node', enumerate_chapters_to_run_node)
    # ])
    builder.add_sequence([
        ('create_folder_dao', wrap_node(create_folder_dao, 'create_folder_dao')),
        ('enumerate_chapters_to_run_node', wrap_node(enumerate_chapters_to_run_node, 'enumerate_chapters_to_run_node'))
        ])

    # We want to run each chapter concurrently.
    # To do that, we want to define a fan-out edge and fan-in node.
    # It needs to be defined in reverse order.
    # Fan-In <- Branches <- Fan-Out

    # Fan In node that will collect all chapters
    # builder.add_node("chapter_summary_completed_node",
    #                  chapter_summary_completed_node)
    builder.add_node("chapter_summary_completed_node", wrap_node(chapter_summary_completed_node, 'chapter_summary_completed_node'))

    # Define Branches:Loop over all chapters and create a node for each.
    for node_name, node in chapter_summary_nodes.items():
        builder.add_node(node_name, node)
        builder.add_edge(node_name, "chapter_summary_completed_node")

    # Fan Out Edge
    builder.add_conditional_edges("enumerate_chapters_to_run_node",
                                  chapter_summary_routing,
                                  path_map=list(chapter_summary_nodes.keys()),
                                  )
    # The end state of above is in the node "chapter_summary_completed_node"
    # That node collects all the branches, but leaves them untouched.

    # Merge the results of all branches
    # builder.add_node("chapter_summary_deep_merge_node",
    #                  chapter_summary_deep_merge_node)
    builder.add_node("chapter_summary_deep_merge_node", wrap_node(chapter_summary_deep_merge_node, 'chapter_summary_deep_merge_node'))
    builder.add_edge("chapter_summary_completed_node",
                     'chapter_summary_deep_merge_node')


    builder.add_edge('chapter_summary_deep_merge_node', 'pydantic_to_dict_node')
    # builder.add_node("pydantic_to_dict_node", pydantic_to_dict_node)
    builder.add_node("pydantic_to_dict_node", wrap_node(pydantic_to_dict_node, 'pydantic_to_dict_node'))

    return builder.compile()


def save_summary_to_json(state):
    """
    Temporary function to save summary results to json. Used for progressive summary testing

    Args:
        state (SummaryState): The graph state for summarization

    Returns:
        None
    """
    import os, json
    if not state.config.timeline_config.progressive_summary_config.archive_summary:
        return
    document_ids = sorted(list(state.folder_dao.documents_df['document_id'].unique()))

    directory = 'archived_summaries_pii'
    filename = f"archived_summary-{state.folder_num}-structured_timeline-{len(document_ids)}.json"

    output_json = {'chapters': [],
                  'chapter_timelines': {},
                  'documentIds': document_ids
                  }
    print(f"Saving Summary! {filename}")
    for chapter, v in state.case_timeline.chapter_timelines.items():
        output_json['chapters'].append(chapter)
        output_json['chapter_timelines'][chapter] = v.structured_timeline


    with open(os.path.join(directory, filename), 'w') as f:
        json.dump(output_json, f, indent=4)

    return
