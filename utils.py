import hashlib

import pymupdf


def chunk_text(text, chunk_size=1200, overlap=200):
    """
    Split text into overlapping chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def get_uploaded_file_signature(uploaded_files):
    """
    Create a signature that changes if uploaded filenames
    or file contents change.
    """
    signatures = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.sha256(
            file_bytes
        ).hexdigest()

        signatures.append(
            (
                uploaded_file.name,
                file_hash,
            )
        )

    return tuple(signatures)


def process_pdfs(pdfs):
    """
    Extract text from stored PDFs and split it into chunks.
    """
    all_chunks = []

    for stored_pdf in pdfs:

        pdf = pymupdf.open(
            stream=stored_pdf["data"],
            filetype="pdf",
        )

        for page_number, page in enumerate(
            pdf,
            start=1,
        ):
            page_text = page.get_text()
            page_chunks = chunk_text(page_text)

            for chunk_number, chunk in enumerate(
                page_chunks,
                start=1,
            ):
                all_chunks.append(
                    {
                        "file_name": stored_pdf["file_name"],
                        "page_number": page_number,
                        "chunk_number": chunk_number,
                        "text": chunk,
                    }
                )

        pdf.close()

    return all_chunks


def create_embedding(text, gemini_client):
    """
    Create an embedding using Gemini.
    """
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )

    return result.embeddings[0].values


def create_chunk_id(chunk):
    """
    Create a deterministic ID for a chunk.
    """
    raw_id = (
        f"{chunk['file_name']}:"
        f"{chunk['page_number']}:"
        f"{chunk['chunk_number']}"
    )

    return hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()


def index_chunks(
    chunks,
    gemini_client,
    collection,
    batch_size=25,
):
    """
    Embed and store chunks in batches.
    """

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        texts = [chunk["text"] for chunk in batch]

        result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
        )

        embeddings = [
            embedding.values
            for embedding in result.embeddings
        ]

        ids = [
            create_chunk_id(chunk)
            for chunk in batch
        ]

        metadatas = [
            {
                "file_name": chunk["file_name"],
                "page_number": chunk["page_number"],
                "chunk_number": chunk["chunk_number"],
            }
            for chunk in batch
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )


def format_chroma_results(results):
    """
    Convert a Chroma result into plain text that can
    be returned to Gemini as a tool result.
    """
    context_parts = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    for document, metadata in zip(
        documents,
        metadatas,
    ):
        context_parts.append(
            f"""
            Source: {metadata["file_name"]}
            Page: {metadata["page_number"]}
            Chunk: {metadata["chunk_number"]}

            {document}
            """.strip()
                    )

    return "\n\n---\n\n".join(context_parts)