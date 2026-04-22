# Memory Buffer Design for Streaming Video-LLM Systems

## Background

Modern Video-LLM systems often operate in streaming scenarios, where video frames arrive sequentially over time rather than as a pre-recorded clip. In such settings, the model must maintain awareness of past visual context in order to correctly answer user queries.

A common architectural component used for this purpose is a **Memory Buffer (MB)** — a module that stores previously observed visual information and allows the system to retrieve relevant context when answering a query.

When a user query `Q` arrives, the system should:

1. Retrieve relevant visual information from the memory buffer.
2. Convert this information into embeddings or tokens.
3. Append them to the LLM input sequence together with the query.
4. Generate the final response.

Your task is to design and implement a memory buffer mechanism for this streaming Video-LLM scenario.

---

## Task Description

You need to design a **Memory Buffer (MB)** capable of storing visual information from a streaming video and retrieving relevant context based on a textual query.

Your solution should address the following aspects.

### 1. Memory Buffer Design

Design a data structure and logic for the Memory Buffer. The design should consider:

- Continuous arrival of video frames

- Limited memory capacity
- The need to preserve useful context over time

You should describe:

- What information is stored in the buffer
- How memory size is controlled
- What policies determine what stays and what gets removed

Explain the motivation behind your design choices.

### 2. Strategy for Storing Video Content

Implement a strategy for efficiently storing information from incoming video frames. The goal is to avoid storing everything while still preserving important context.

The system should process a continuous stream of frames and update the memory buffer accordingly.

### 3. Query-Based Memory Retrieval

When a user submits a text query `Q`, the system must retrieve the most relevant information from the memory buffer.

Design and implement a retrieval mechanism that:

- Converts the query into an embedding
- Searches for relevant visual content in memory
- Returns a subset of memory entries to be used as context for the LLM

Explain how your retrieval method works and why it is suitable for streaming video understanding.

### 4. Integration with LLM Input

Demonstrate how retrieved memory entries would be integrated into the LLM input sequence.

Describe:

- What tokens/embeddings are passed to the LLM
- How the retrieved memory is formatted
- How the query and visual context are combined

A simplified simulation is sufficient — a full LLM implementation is not required.

---

## Deliverables

Submit:

- A **code implementation in a Jupyter notebook**
- **Rationales as Markdown cells** for your design decisions, explaining:
  - Memory structure
  - Storage strategy
  - Retrieval method
  - Design trade-offs

---

## Notes

- Implementing full response generation with a VLM/LLM is **not required**. It is sufficient to use a visual encoder and a text encoder to produce embeddings.
- Please prefer **smaller models** in terms of parameter count to keep the solution lightweight and computationally efficient.

---

## Example Usage

- Simulate a stream of frames (you can load a long video, e.g. 1h+)
- Insert them into memory
- Run several example queries
- Show retrieved memory entries