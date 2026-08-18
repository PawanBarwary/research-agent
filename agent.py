from google.genai import types

from utils import (
    create_embedding,
    format_chroma_results,
)


def log_tool(tool_log, name, **kwargs):
    tool_log.append({
        "name": name,
        "args": kwargs,
    })


def run_research_agent(
    question,
    gemini_client,
    collection,
    pdfs,
):
    """
    Run a lightweight research agent with retrieval tools.
    """

    tool_log = []

    available_files = [
        pdf["file_name"]
        for pdf in pdfs
    ]

    # -----------------------------------------------------
    # Tool 1: Search one paper
    # -----------------------------------------------------

    def search_one_paper(
        file_name: str,
        query: str,
        n_results: int = 3,
    ) -> str:
        """
        Search for passages in one specific research paper.

        Args:
            file_name:
                Exact filename of the paper to search.

            query:
                Semantic search query describing the
                information needed.

            n_results:
                Number of passages to retrieve.

        Returns:
            Relevant passages with page-level source metadata.
        """

        log_tool(
            tool_log,
            "search_one_paper",
            file_name=file_name,
            query=query,
        )


        if file_name not in available_files:
            return (
                f"Unknown paper '{file_name}'. "
                f"Available papers: {available_files}"
            )

        query_embedding = create_embedding(
            query,
            gemini_client,
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={
                "file_name": file_name
            },
        )

        return format_chroma_results(results)


    # -----------------------------------------------------
    # Tool 2: Search every paper
    # -----------------------------------------------------

    def search_all_papers(
        query: str,
        n_results_per_paper: int = 2,
    ) -> str:
        """
        Search all uploaded research papers.

        Use this when the question requires comparing papers
        or when it is unclear which paper contains the answer.

        Args:
            query:
                Semantic search query.

            n_results_per_paper:
                Number of passages to retrieve from each paper.

        Returns:
            Relevant passages from all papers with source metadata.
        """

        log_tool(
            tool_log,
            "search_all_papers",
            query=query,
        )

        all_context = []

        query_embedding = create_embedding(
            query,
            gemini_client,
        )

        for file_name in available_files:

            results = collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=n_results_per_paper,
                where={
                    "file_name": file_name
                },
            )

            paper_context = format_chroma_results(
                results
            )

            all_context.append(
                paper_context
            )

        return "\n\n========\n\n".join(
            all_context
        )


    # -----------------------------------------------------
    # Tool 3: Retrieve additional page context
    # -----------------------------------------------------

    def get_page_context(
        file_name: str,
        page_number: int,
    ) -> str:
        """
        Retrieve all indexed chunks from a specific page.

        Use this when an initially retrieved passage appears
        relevant but more surrounding context is needed.

        Args:
            file_name:
                Exact filename of the paper.

            page_number:
                Page number to retrieve.

        Returns:
            All indexed text chunks from that page.
        """

        log_tool(
            tool_log,
            "get_page_context",
            file_name=file_name,
            page_number=page_number,
        )


        if file_name not in available_files:
            return (
                f"Unknown paper '{file_name}'. "
                f"Available papers: {available_files}"
            )

        results = collection.get(
            where={
                "$and": [
                    {
                        "file_name": {
                            "$eq": file_name
                        }
                    },
                    {
                        "page_number": {
                            "$eq": page_number
                        }
                    },
                ]
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

        if not results["documents"]:
            return (
                f"No indexed text found for "
                f"{file_name}, page {page_number}."
            )

        combined = []

        for document, metadata in zip(
            results["documents"],
            results["metadatas"],
        ):
            combined.append(
                f"""
                Source: {metadata["file_name"]}
                Page: {metadata["page_number"]}
                Chunk: {metadata["chunk_number"]}

                {document}
                """.strip()
            )

        return "\n\n---\n\n".join(
            combined
        )


    # -----------------------------------------------------
    # Agent instructions
    # -----------------------------------------------------

    system_instruction = f"""
        You are a research assistant with access to retrieval tools
        over the user's uploaded academic papers.

        Available papers:
        {available_files}

        Your job is to answer research questions accurately using
        evidence from the uploaded papers.

        Use the available retrieval tools whenever factual evidence
        from the papers is required.

        Guidelines:

        1. Use search_one_paper when the question clearly concerns
        one specific paper.

        2. Use search_all_papers when:
        - the user asks for a comparison,
        - multiple papers are relevant,
        - or you do not know which paper contains the answer.

        3. Use get_page_context when an initial passage is relevant
        but insufficient and you need additional surrounding
        evidence.

        4. Do not invent claims that are not supported by retrieved
        evidence.

        5. If the available papers do not contain enough information,
        clearly say so.

        6. Cite factual claims using:
        [filename, p. X]

        7. Give concise but technically useful answers.
        """


    # -----------------------------------------------------
    # Let Gemini choose and execute tools
    # -----------------------------------------------------

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[
                search_one_paper,
                search_all_papers,
                get_page_context,
            ],
            automatic_function_calling=(
                types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=10
                )
            ),
        ),
    )

    return response, tool_log