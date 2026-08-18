"""
Sample document corpus for the hybrid RAG gateway.
25 realistic technical documents across distributed systems, vector databases,
LLM infrastructure, RAG systems, Kubernetes, API design, database indexing, and ML pipelines.
"""

DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Consistent Hashing in Distributed Systems",
        "source": "distributed-systems/consistent-hashing.md",
        "text": (
            "Consistent hashing is a distributed hashing scheme that operates independently of the number "
            "of servers or objects in a distributed hash table by assigning them a position on an abstract "
            "circle or hash ring. This allows servers and objects to scale without affecting the overall system. "
            "When a node is added or removed, only a fraction of the keys need to be remapped — specifically, "
            "only the keys that were mapped to the removed node are redistributed. In contrast, with a naive "
            "modular hashing approach, almost every key would need to be remapped when the cluster size changes. "
            "Virtual nodes are a common enhancement: each physical server is represented by multiple points on "
            "the ring, improving load distribution. Cassandra and DynamoDB both use consistent hashing as their "
            "core partitioning strategy. The trade-off is implementation complexity versus operational flexibility. "
            "Production deployments typically use 150-200 virtual nodes per physical machine to achieve near-uniform "
            "load distribution. Hotspot mitigation requires careful monitoring of key access patterns, and "
            "replication across adjacent ring segments provides fault tolerance when nodes become unavailable."
        ),
    },
    {
        "id": "doc_002",
        "title": "FAISS: Efficient Similarity Search at Scale",
        "source": "vector-databases/faiss-overview.md",
        "text": (
            "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering "
            "of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones "
            "that do not fit in RAM. FAISS is written in C++ with complete wrappers for Python. It uses SIMD "
            "intrinsics and GPU acceleration where available. The key index types are Flat (exact brute-force), "
            "IVF (inverted file index for approximate search), HNSW (hierarchical navigable small world graphs), "
            "and PQ (product quantization for compression). IVF + PQ is the most common production configuration, "
            "trading recall for speed and memory. For 10M vectors of dimension 768, an IVFFlat index can search "
            "in under 10ms with 95%+ recall at nprobe=64. Indexing throughput scales linearly with CPU cores. "
            "Choosing the right number of centroids (nlist) is critical: sqrt(n) is a good starting point, "
            "typically 4096 to 65536 for large corpora. GPU-accelerated FAISS achieves 5-10x speedup over CPU "
            "for batch queries, making it ideal for high-throughput embedding search pipelines."
        ),
    },
    {
        "id": "doc_003",
        "title": "BM25: The Gold Standard for Lexical Retrieval",
        "source": "information-retrieval/bm25-deep-dive.md",
        "text": (
            "BM25 (Best Match 25) is a probabilistic relevance function used by search engines to rank documents "
            "given a user's query. It is a bag-of-words retrieval function that ranks documents based on the query "
            "terms appearing in each document, regardless of their proximity within the document. BM25 extends TF-IDF "
            "by introducing two key parameters: k1, which controls term frequency saturation (typically 1.2-2.0), "
            "and b, which controls document length normalization (typically 0.75). The saturation prevents very "
            "frequent terms from dominating the score, which is a known weakness of raw TF-IDF. BM25 remains "
            "highly competitive in benchmarks like BEIR, often outperforming neural methods on keyword-heavy queries. "
            "Hybrid retrieval systems typically combine BM25 with dense vector search using score fusion techniques "
            "like Reciprocal Rank Fusion. Elasticsearch's default similarity function is BM25F, a field-weighted "
            "variant. BM25 is particularly strong on technical documentation where exact term matches (function names, "
            "error codes, configuration keys) are critical signals that dense models may miss due to tokenization."
        ),
    },
    {
        "id": "doc_004",
        "title": "Reciprocal Rank Fusion for Hybrid Search",
        "source": "information-retrieval/rrf-fusion.md",
        "text": (
            "Reciprocal Rank Fusion (RRF) is a rank aggregation method for combining multiple ranked lists of "
            "documents. For each document, RRF computes a combined score as the sum of reciprocal ranks across "
            "all input rankers: RRF(d) = sum(1 / (k + r_i(d))), where r_i(d) is the rank of document d in "
            "ranker i and k is a constant (typically 60) that smooths the impact of high-ranking documents. "
            "RRF was introduced by Cormack, Clarke, and Buettcher in 2009 and consistently outperforms linear "
            "score interpolation in practice because it is robust to score scale differences between rankers. "
            "For hybrid RAG systems combining BM25 and dense vector search, RRF provides a parameter-free fusion "
            "that avoids the need to tune weighting coefficients. The k=60 default means that rank 1 contributes "
            "1/61 ≈ 0.0164, rank 10 contributes 1/70 ≈ 0.0143, and rank 100 contributes 1/160 ≈ 0.00625. "
            "This gentle decay ensures documents ranked highly by any individual retriever receive credit, "
            "while heavily penalizing documents that rank poorly across all methods. RRF is used in production "
            "at Vespa, Elasticsearch 8.x, and several commercial RAG platforms."
        ),
    },
    {
        "id": "doc_005",
        "title": "Cross-Encoder Reranking for Neural IR",
        "source": "information-retrieval/cross-encoder-reranking.md",
        "text": (
            "Cross-encoders are neural models that jointly encode a query and a document together, producing "
            "a relevance score. Unlike bi-encoders (used in dense retrieval) which encode query and document "
            "independently, cross-encoders have full attention over both texts simultaneously, giving them "
            "superior relevance modeling at the cost of much higher inference latency. The typical pipeline is: "
            "fast first-stage retrieval (BM25 or ANN) to fetch 100-200 candidates, then cross-encoder reranking "
            "to select the top 5-10. Cross-encoders based on BERT score around 88+ MRR@10 on MS MARCO, vs "
            "~70 MRR@10 for bi-encoders. The Sentence Transformers library provides cross-encoder models fine-tuned "
            "on MS MARCO: ms-marco-MiniLM-L-6-v2 (fast, 80ms per batch of 32) and ms-marco-electra-base (slower, "
            "higher accuracy). Key hyperparameters: batch size for parallel scoring, maximum sequence length "
            "(typically 512 tokens), and the number of candidates to rerank. Production rerankers often use "
            "ONNX Runtime or TensorRT quantization to achieve 4-8x throughput improvement while maintaining "
            "within 1% of full-precision accuracy."
        ),
    },
    {
        "id": "doc_006",
        "title": "Semantic Caching for LLM Inference Cost Reduction",
        "source": "llm-infrastructure/semantic-cache.md",
        "text": (
            "Semantic caching stores previous LLM responses and retrieves them when a semantically similar query "
            "arrives, bypassing the expensive inference call. Unlike exact-match caching, semantic caching uses "
            "embedding similarity to match queries, typically with a cosine similarity threshold of 0.92-0.97. "
            "Setting the threshold too low causes incorrect cache hits (different questions get the same answer); "
            "too high and the cache hit rate becomes negligible. A threshold of 0.95-0.96 is common in practice. "
            "The cache key is the query embedding vector; the value is the full response. Redis with the RediSearch "
            "module supports vector similarity search natively, enabling sub-millisecond cache lookups even at "
            "millions of stored entries. GPTCache is an open-source library that implements this pattern with "
            "pluggable backends. In production, semantic caches reduce LLM API costs by 30-60% on typical enterprise "
            "Q&A workloads where users repeatedly ask similar questions. Cache warming with common queries and "
            "TTL-based expiry (24-72 hours) are standard operational practices. Monitoring cache hit rate and "
            "hit quality (via human evaluation sampling) are essential to catch threshold drift over time."
        ),
    },
    {
        "id": "doc_007",
        "title": "Pinecone Vector Database Architecture",
        "source": "vector-databases/pinecone-architecture.md",
        "text": (
            "Pinecone is a fully managed vector database optimized for machine learning applications. Its core "
            "architecture separates storage and compute: vectors are stored in a distributed object store while "
            "a serving layer maintains an approximate nearest neighbor (ANN) index in memory for low-latency queries. "
            "Pinecone uses a custom HNSW variant tuned for cloud deployment, with automatic index rebuilding when "
            "new vectors are upserted. Namespaces provide logical partitioning within an index, enabling multi-tenant "
            "architectures without index duplication overhead. Metadata filtering runs in parallel with vector search "
            "using a pre-filtering strategy that first restricts the search space by metadata predicates before ANN. "
            "This avoids post-filtering recall degradation common with large filter ratios. Pinecone's serverless "
            "tier separates read and write paths, allowing independent scaling. Write throughput is limited by index "
            "rebuild frequency; read QPS scales horizontally with replicas. The gRPC client library achieves "
            "significantly lower P99 latency than REST due to connection reuse and binary framing. For RAG systems, "
            "Pinecone's performance characteristics make it suitable for corpora of 10M to 1B vectors."
        ),
    },
    {
        "id": "doc_008",
        "title": "RAG System Design: Chunking Strategies",
        "source": "rag-systems/chunking-strategies.md",
        "text": (
            "Document chunking is one of the highest-leverage decisions in RAG system design. The chunk size "
            "determines the granularity of retrieval: small chunks (128-256 tokens) retrieve precise passages "
            "but lose surrounding context; large chunks (512-1024 tokens) preserve context but dilute relevance "
            "signals. Sentence-level chunking with overlap (typically 20-30% overlap) is a strong baseline. "
            "Semantic chunking uses embedding similarity between adjacent sentences to find natural breakpoints, "
            "outperforming fixed-size chunking on heterogeneous documents. Hierarchical chunking stores both "
            "sentence-level and paragraph-level embeddings, using small chunks for retrieval but returning parent "
            "chunks to the LLM for broader context — this is the 'small-to-big' retrieval pattern. For code "
            "documentation, AST-aware chunking splits at function and class boundaries. Token budget management "
            "is critical: with a 4K context LLM and 5 retrieved chunks, each chunk should be at most 400 tokens "
            "to leave room for the query and system prompt. Chunk IDs should encode position metadata (document ID, "
            "section, paragraph index) to enable citation and deduplication of overlapping chunks."
        ),
    },
    {
        "id": "doc_009",
        "title": "Kubernetes HPA and VPA for ML Workloads",
        "source": "kubernetes/autoscaling-ml-workloads.md",
        "text": (
            "Horizontal Pod Autoscaler (HPA) and Vertical Pod Autoscaler (VPA) serve complementary roles in "
            "scaling ML inference services on Kubernetes. HPA scales the number of pod replicas based on observed "
            "metrics — CPU, memory, or custom metrics like requests-per-second from Prometheus. VPA adjusts "
            "individual pod resource requests and limits based on historical usage, right-sizing pods that are "
            "over- or under-provisioned. For GPU-based inference, KEDA (Kubernetes Event Driven Autoscaler) is "
            "preferred over native HPA because it can scale on queue depth (e.g., SQS or Kafka lag), enabling "
            "scale-to-zero. GPU nodes use taints and node selectors to prevent non-GPU workloads from consuming "
            "expensive GPU capacity. Model loading latency is the primary cold-start bottleneck: pre-loading "
            "models into shared memory or using init containers to warm up replicas before they enter the "
            "service mesh reduces P99 latency spikes during scale-out events. Resource quotas at the namespace "
            "level prevent runaway scaling from exhausting cluster capacity. Bin-packing with pod affinity rules "
            "improves GPU utilization by co-locating complementary workloads on the same node."
        ),
    },
    {
        "id": "doc_010",
        "title": "API Rate Limiting: Token Bucket and Sliding Window",
        "source": "api-design/rate-limiting-algorithms.md",
        "text": (
            "Rate limiting protects APIs from overload and abuse. The two dominant algorithms are token bucket "
            "and sliding window log. Token bucket maintains a counter that refills at a fixed rate; each request "
            "consumes one token, and requests are rejected when the bucket is empty. It allows short bursts up "
            "to the bucket capacity, making it suitable for APIs where bursty traffic is acceptable. Sliding "
            "window log maintains a timestamped log of requests within the last N seconds; it provides precise "
            "rate limiting without burst allowance but uses O(requests) memory. A compromise is the sliding "
            "window counter, which approximates the sliding log using two fixed windows and a weighted average — "
            "memory-efficient and accurate to within ~0.1% for typical traffic distributions. Redis is the "
            "standard backend for distributed rate limiters; Lua scripts ensure atomicity of check-and-increment "
            "operations. HTTP 429 Too Many Requests with Retry-After header is the standard response. "
            "Hierarchical rate limits (per-IP, per-user, per-API-key, global) layered in order prevent any "
            "single vector of abuse from exhausting shared capacity. Exponential backoff in clients combined "
            "with jitter prevents the thundering herd problem when limits reset."
        ),
    },
    {
        "id": "doc_011",
        "title": "PostgreSQL Index Types: B-Tree, GIN, and HNSW",
        "source": "databases/postgresql-index-types.md",
        "text": (
            "PostgreSQL supports multiple index types suited to different access patterns. B-Tree is the default "
            "and handles equality, range queries, and sorting efficiently with O(log n) lookup. GIN (Generalized "
            "Inverted Index) excels at indexing composite values like arrays, JSONB, and full-text search vectors; "
            "it stores a mapping from each element to the set of rows containing it, enabling fast containment "
            "queries. BRIN (Block Range INdex) is extremely small and efficient for naturally ordered columns "
            "like timestamps in append-only tables, but only suitable when physical ordering correlates with "
            "logical ordering. PostgreSQL 16 introduced native HNSW and IVFFlat index types via the pgvector "
            "extension, enabling vector similarity search directly in SQL. HNSW offers better query performance "
            "while IVFFlat has faster index build time. Partial indexes (WHERE clause on index) dramatically "
            "reduce index size for selective queries — e.g., indexing only active records. Expression indexes "
            "on functions like lower(email) enable case-insensitive lookups without full table scans. Index "
            "bloat from high-churn tables requires periodic REINDEX or autovacuum tuning to maintain performance."
        ),
    },
    {
        "id": "doc_012",
        "title": "MLflow for Experiment Tracking and Model Registry",
        "source": "ml-pipelines/mlflow-experiment-tracking.md",
        "text": (
            "MLflow is an open-source platform for managing the ML lifecycle, including experiment tracking, "
            "model packaging, and deployment. The Tracking component logs parameters, metrics, and artifacts "
            "for each training run, storing them in a backend store (SQLite for local, PostgreSQL for production) "
            "and an artifact store (local filesystem, S3, GCS). The Model Registry provides versioning, stage "
            "transitions (Staging → Production → Archived), and access control. MLflow's autologging feature "
            "automatically captures hyperparameters and metrics for scikit-learn, PyTorch, and TensorFlow "
            "without manual instrumentation. The mlflow.pyfunc flavor provides a generic Python function "
            "interface for serving any model, enabling consistent deployment regardless of the underlying "
            "framework. For RAG pipelines, tracking embedding model versions alongside retriever configurations "
            "is essential for reproducibility — a change in embedding model invalidates all cached vectors. "
            "MLflow Projects define reproducible training environments using conda or Docker. Integration with "
            "Kubernetes via the MLflow Projects Kubernetes backend enables distributed training with automatic "
            "resource allocation and job monitoring through the MLflow UI."
        ),
    },
    {
        "id": "doc_013",
        "title": "Raft Consensus Algorithm for Distributed Coordination",
        "source": "distributed-systems/raft-consensus.md",
        "text": (
            "Raft is a consensus algorithm designed to be more understandable than Paxos while providing "
            "equivalent correctness guarantees. A Raft cluster consists of one leader and multiple followers. "
            "The leader receives all client writes, appends them to its log, and replicates to followers. "
            "A write is committed once a majority (quorum) of nodes acknowledge it. If the leader fails, "
            "followers hold an election: any node that hasn't received a heartbeat within the election timeout "
            "increments its term, votes for itself, and requests votes from peers. The first candidate to "
            "receive a majority wins and becomes the new leader. Log matching property guarantees that if two "
            "logs have the same index and term for an entry, they are identical up to that entry. Safety is "
            "guaranteed by requiring candidates to have a log at least as up-to-date as any majority member. "
            "Raft is used in etcd (Kubernetes' backing store), CockroachDB, TiKV, and Consul. Typical "
            "election timeouts are 150-300ms; heartbeat intervals 50-100ms. Leader leases can serve reads "
            "without log roundtrips, reducing read latency at the cost of weaker consistency during network "
            "partitions."
        ),
    },
    {
        "id": "doc_014",
        "title": "Transformer Attention Mechanisms: MHA and GQA",
        "source": "llm-infrastructure/attention-mechanisms.md",
        "text": (
            "Multi-Head Attention (MHA) is the core mechanism of transformer models, allowing each token to "
            "attend to all other tokens in the sequence. Each head learns a different subspace projection of "
            "the query, key, and value matrices, capturing different syntactic and semantic relationships. "
            "For a model with d_model=4096 and 32 heads, each head operates on 128-dimensional projections. "
            "The attention computation is O(n²) in sequence length, making long-context inference expensive. "
            "Grouped Query Attention (GQA), used in Llama 2 70B and Mistral, reduces the number of KV heads "
            "while keeping query heads constant — e.g., 32 query heads sharing 8 KV heads. This reduces the "
            "KV cache memory footprint by 4x with minimal quality degradation, enabling longer context and "
            "larger batch sizes. Multi-Query Attention (MQA) takes this further with a single KV head, used "
            "in Falcon and PaLM. Flash Attention implements the attention kernel with tiling to avoid "
            "materializing the full attention matrix in HBM, achieving 2-4x speedup and enabling 4-8x longer "
            "sequences within the same GPU memory budget. FlashAttention-2 further optimizes thread block "
            "partitioning for better GPU occupancy."
        ),
    },
    {
        "id": "doc_015",
        "title": "Redis Pub/Sub and Streams for Real-Time Pipelines",
        "source": "databases/redis-pubsub-streams.md",
        "text": (
            "Redis provides two mechanisms for real-time messaging: Pub/Sub and Streams. Pub/Sub is a fire-and-forget "
            "broadcast where subscribers receive messages published after they subscribe — messages published to "
            "offline subscribers are lost. It suits ephemeral notifications but not durable event queues. Redis "
            "Streams (introduced in Redis 5.0) is a persistent, append-only log data structure similar to Kafka. "
            "Streams support consumer groups where multiple consumers can process messages cooperatively with "
            "at-least-once delivery guarantees and explicit acknowledgment. Stream entries have auto-generated "
            "IDs combining millisecond timestamp and sequence number. XADD adds entries; XREADGROUP reads with "
            "consumer group semantics; XACK acknowledges processing. Pending entries list (PEL) tracks "
            "unacknowledged messages for redelivery on consumer failure. For RAG pipelines, Streams enable "
            "asynchronous document ingestion: an ingest service publishes documents to a stream, and embedding "
            "workers consume from consumer groups in parallel. MAXLEN with the ~ approximate trimming option "
            "keeps stream size bounded without expensive exact-count overhead. Keyspace notifications allow "
            "monitoring cache expiration events for semantic cache TTL management."
        ),
    },
    {
        "id": "doc_016",
        "title": "Sentence Transformers: Dense Embedding Models",
        "source": "llm-infrastructure/sentence-transformers.md",
        "text": (
            "Sentence Transformers is a Python framework for generating dense vector representations of sentences "
            "and paragraphs using transformer models fine-tuned with contrastive learning objectives. The "
            "all-MiniLM-L6-v2 model produces 384-dimensional embeddings at ~14,000 sentences/second on CPU, "
            "making it practical for real-time embedding in RAG pipelines. all-mpnet-base-v2 produces "
            "768-dimensional embeddings with higher quality but 3x lower throughput. For domain-specific "
            "applications, fine-tuning on in-domain (query, positive, negative) triplets using MultipleNegatives "
            "RankingLoss substantially improves retrieval performance over zero-shot models. The SBERT training "
            "objective uses mean pooling over the last hidden states of the [CLS] and all token embeddings, "
            "then L2-normalizes for cosine similarity. Asymmetric semantic search uses separate models for "
            "queries (short, keyword-like) and passages (long, informational), trained with bi-encoder "
            "distillation from a cross-encoder teacher. Batch encoding with GPU acceleration achieves 10-50x "
            "throughput improvement over single-sample inference. Model outputs should be L2-normalized before "
            "storage to enable inner product as a proxy for cosine similarity, enabling FAISS IndexFlatIP "
            "instead of the slower IndexFlatL2."
        ),
    },
    {
        "id": "doc_017",
        "title": "gRPC vs REST for Microservice Communication",
        "source": "api-design/grpc-vs-rest.md",
        "text": (
            "gRPC and REST represent fundamentally different approaches to inter-service communication. REST "
            "uses HTTP/1.1 with JSON, offering human-readable payloads, broad tooling support, and natural "
            "alignment with web browsers. gRPC uses HTTP/2 with Protocol Buffers, providing binary framing, "
            "multiplexed streams, header compression, and strongly-typed service contracts via .proto files. "
            "Benchmarks consistently show gRPC achieving 5-10x higher throughput and 2-3x lower latency than "
            "REST for equivalent payloads, primarily due to Protocol Buffer serialization efficiency and "
            "HTTP/2 connection reuse. gRPC supports four communication patterns: unary RPC (request-response), "
            "server streaming, client streaming, and bidirectional streaming — the latter enabling real-time "
            "inference pipelines where tokens stream back as they are generated. Service mesh integration "
            "(Istio, Linkerd) works seamlessly with gRPC, providing observability and traffic management. "
            "The main drawback is browser incompatibility; gRPC-Web provides a browser-compatible subset via "
            "an Envoy proxy. For RAG systems, gRPC is preferred for embedding service calls (high throughput) "
            "while REST or SSE is preferred for the user-facing API (browser compatibility, streaming events)."
        ),
    },
    {
        "id": "doc_018",
        "title": "Vector Quantization and Product Quantization",
        "source": "vector-databases/product-quantization.md",
        "text": (
            "Product Quantization (PQ) is a compression technique for high-dimensional vectors that dramatically "
            "reduces memory footprint while preserving approximate nearest-neighbor query capability. PQ splits "
            "each vector into M subvectors of equal dimension d/M, then quantizes each subvector independently "
            "using a learned codebook of K codewords (typically K=256). Each subvector is replaced by an 8-bit "
            "index into its codebook, reducing a 768-dimensional float32 vector (3072 bytes) to M=96 bytes — "
            "a 32x compression ratio. Distance computation uses precomputed lookup tables: for each query "
            "subvector, compute distances to all K codewords once, then the approximate distance to any "
            "database vector is a lookup and sum. This asymmetric distance computation enables searching "
            "1M vectors in under 1ms on CPU. IVFPQ combines an inverted file index with PQ: the IVF coarsely "
            "partitions the space into nlist Voronoi cells, and PQ compresses residuals within each cell. "
            "Scalar quantization (SQ8) is a simpler alternative that quantizes each float32 to int8, achieving "
            "4x compression with minimal accuracy loss and SIMD-accelerated distance computation. OPQ (Optimized "
            "PQ) applies a rotation matrix before PQ to improve compression quality, especially for correlated dimensions."
        ),
    },
    {
        "id": "doc_019",
        "title": "LLM Context Window Management and KV Cache",
        "source": "llm-infrastructure/context-window-kv-cache.md",
        "text": (
            "The KV cache is the core data structure enabling efficient autoregressive generation in transformer "
            "models. During prefill, the model processes all input tokens and caches their key and value "
            "projections for each attention head and layer. During decoding, each new token only needs to attend "
            "to cached KVs, avoiding redundant computation. KV cache size is: 2 × num_layers × num_kv_heads × "
            "head_dim × seq_len × bytes_per_element. For Llama 2 70B with FP16, this is ~2.6GB per 4K token "
            "request — the primary memory constraint on batch size. PagedAttention (vLLM) manages KV cache "
            "with non-contiguous memory blocks, similar to OS virtual memory, enabling near-zero memory waste "
            "and 3-4x higher throughput vs naive static allocation. Prompt caching (Anthropic, OpenAI) extends "
            "this to the API level: repeated system prompts across requests reuse cached KVs, reducing time-to-"
            "first-token and cost for long shared prefixes. Chunked prefill interleaves prefill and decode phases "
            "to reduce time-to-first-token for long prompts without sacrificing decode throughput. Speculative "
            "decoding uses a small draft model to propose tokens that a large model verifies in parallel, "
            "achieving 2-3x decoding speedup for high-quality draft models."
        ),
    },
    {
        "id": "doc_020",
        "title": "Apache Kafka for High-Throughput Event Streaming",
        "source": "distributed-systems/kafka-event-streaming.md",
        "text": (
            "Apache Kafka is a distributed event streaming platform designed for high-throughput, fault-tolerant, "
            "and ordered event processing. Kafka organizes data into topics, each partitioned across brokers "
            "for parallelism. Producers append records to partition leaders; consumer groups read from partitions "
            "with each partition assigned to exactly one consumer per group, providing ordered consumption within "
            "a partition. Replication factor (typically 3) ensures durability: the ISR (in-sync replicas) set "
            "determines which replicas are eligible for leader election. acks=all requires acknowledgment from "
            "all ISR members, providing strongest durability at the cost of latency. Compacted topics retain "
            "only the latest record per key, enabling Kafka to serve as a changelog store for derived views. "
            "Kafka Streams and ksqlDB enable stateful stream processing without external databases. Retention "
            "policies are time-based or size-based; tiered storage offloads old segments to object storage "
            "while maintaining fast access to recent data. For ML pipelines, Kafka serves as the backbone for "
            "feature pipelines, model monitoring event streams, and online serving logs that feed back into "
            "training data. Consumer lag monitoring via JMX or Kafka Exporter is essential for SLA adherence."
        ),
    },
    {
        "id": "doc_021",
        "title": "Observability for ML Systems: Metrics, Logs, Traces",
        "source": "ml-pipelines/ml-observability.md",
        "text": (
            "Observability for ML systems extends beyond traditional software metrics to include model-specific "
            "signals: prediction distributions, feature drift, embedding drift, and business outcome metrics. "
            "The three pillars — metrics, logs, and traces — each serve distinct purposes. Prometheus metrics "
            "capture aggregate statistics (request rate, latency percentiles, error rate, GPU utilization) "
            "with configurable scrape intervals. Structured logs (JSON, preferably) capture per-request context "
            "including query text, retrieved document IDs, reranking scores, and token counts. Distributed "
            "traces via OpenTelemetry link spans across services — embedding service, retriever, reranker, "
            "LLM API — providing end-to-end latency attribution. For RAG systems specifically, logging retrieved "
            "document scores alongside user feedback enables offline retrieval quality analysis. Embedding drift "
            "detection compares the distribution of incoming query embeddings against a reference window using "
            "Maximum Mean Discrepancy or population stability index. Alert thresholds for P99 latency "
            "(typically 2-3s for RAG), cache hit rate drops, and retrieval score distribution shifts trigger "
            "investigation workflows. Grafana dashboards with a golden signals overview (latency, traffic, "
            "errors, saturation) paired with ML-specific panels provide comprehensive operational visibility."
        ),
    },
    {
        "id": "doc_022",
        "title": "Docker Multi-Stage Builds for Python ML Applications",
        "source": "kubernetes/docker-multistage-ml.md",
        "text": (
            "Multi-stage Docker builds significantly reduce final image size for Python ML applications by "
            "separating build-time dependencies from runtime requirements. A typical pattern uses a builder "
            "stage with full development toolchain to compile packages with C extensions (numpy, faiss-cpu, "
            "PyTorch), then copies only the compiled artifacts to a slim runtime stage. Starting from "
            "python:3.11-slim (90MB) instead of python:3.11 (900MB) reduces attack surface. Compiling "
            "dependencies to wheels in the builder stage and installing from local wheels in the runtime stage "
            "avoids build tool installation in the final image. For GPU workloads, nvidia/cuda base images "
            "provide the necessary CUDA toolkit; using the -runtime variant instead of -devel saves ~3GB. "
            "Layer caching is critical for build speed: copy requirements.txt and install dependencies before "
            "copying application code, so code changes don't invalidate the dependency layer. Bind mounts in "
            "BuildKit (--mount=type=cache) cache pip download directories across builds, reducing download time "
            "for iterative development. COPY --chown sets file ownership in the same layer as the copy, "
            "avoiding a separate RUN chown layer. Non-root USER directives improve security posture without "
            "impacting application functionality for network services."
        ),
    },
    {
        "id": "doc_023",
        "title": "Embedding Model Fine-Tuning for Domain Adaptation",
        "source": "llm-infrastructure/embedding-fine-tuning.md",
        "text": (
            "Off-the-shelf embedding models often underperform on specialized domains due to vocabulary mismatch "
            "and concept distribution differences from the pretraining corpus. Fine-tuning on domain-specific "
            "data can improve retrieval MRR by 15-40% relative. The standard approach uses contrastive learning "
            "with in-batch negatives: for each (query, positive_passage) pair in a batch of B, the model must "
            "score the positive higher than B-1 negatives. Hard negative mining — retrieving passages that are "
            "lexically similar but semantically incorrect — is critical; random negatives are too easy and "
            "produce models that still fail on challenging cases. MNRL (Multiple Negatives Ranking Loss) with "
            "temperature scaling (τ=0.05) is the standard loss function. Training data construction uses LLM "
            "generation to produce query variations for each document chunk, providing scalable supervision "
            "without human annotation. Domain adaptation also benefits from continued masked language model "
            "pretraining on domain text before the contrastive fine-tuning stage. Evaluation uses domain-specific "
            "BEIR-style benchmark splits with NDCG@10, MRR@10, and Recall@100. Adapter-based fine-tuning "
            "(LoRA or bottleneck adapters) reduces the number of trainable parameters by 100x, enabling "
            "fine-tuning on a single consumer GPU in hours rather than days."
        ),
    },
    {
        "id": "doc_024",
        "title": "OpenAPI Specification and API Versioning Strategies",
        "source": "api-design/openapi-versioning.md",
        "text": (
            "The OpenAPI Specification (OAS) defines a standard, language-agnostic interface description for "
            "REST APIs. FastAPI auto-generates OpenAPI 3.1 specs from Python type hints and Pydantic models, "
            "providing Swagger UI and ReDoc documentation with zero configuration. API versioning strategies "
            "each involve trade-offs: URL path versioning (/v1/, /v2/) is explicit and cacheable but requires "
            "clients to update base URLs; header versioning (Accept: application/vnd.api+json;version=2) is "
            "cleaner but harder to test in browsers; query parameter versioning (?version=2) is simple but "
            "pollutes query strings and can cause caching issues. Semantic versioning for APIs uses MAJOR.MINOR "
            "where MAJOR increments indicate breaking changes and MINOR indicates additive changes. Sunset headers "
            "(RFC 8594) provide deprecation warnings with a planned removal date, giving clients time to migrate. "
            "API evolution over breaking versioning is preferred: additive changes (new optional fields, new "
            "endpoints) are non-breaking; field removal and type changes require new versions. Schema validation "
            "on input prevents bad data from reaching business logic. Response envelopes with metadata fields "
            "(request_id, timestamp, version) aid debugging and observability without breaking consumers."
        ),
    },
    {
        "id": "doc_025",
        "title": "Hybrid Cloud Architecture for AI Inference",
        "source": "distributed-systems/hybrid-cloud-ai.md",
        "text": (
            "Hybrid cloud architectures for AI inference split workloads across on-premises GPU clusters and "
            "cloud-based inference endpoints based on latency requirements, data sovereignty constraints, and "
            "cost optimization. Latency-sensitive inference (interactive chat, real-time recommendations) runs "
            "on co-located on-prem hardware to avoid network round-trip overhead. Batch inference and model "
            "training exploit cloud elasticity with spot/preemptible instances, reducing cost by 60-80% vs "
            "on-demand. Data locality is the primary driver for on-prem placement: regulated data (PII, PHI, "
            "financial records) that cannot leave the datacenter flows to on-prem models, while public or "
            "de-identified data can use cloud endpoints. A hybrid gateway routes requests to the appropriate "
            "backend using policy rules (data classification, user tier, load) with automatic failover. "
            "Network throughput between on-prem and cloud (AWS Direct Connect, Azure ExpressRoute) must "
            "accommodate model artifact transfers during upgrades. Cost attribution across environments "
            "requires tagging discipline and unified billing dashboards. Edge inference (device-side models) "
            "adds a third tier for offline capability and sub-5ms latency requirements. Model compression "
            "(quantization, pruning, distillation) is typically required to fit capable models within edge "
            "device constraints."
        ),
    },
    {
        "id": "doc_026",
        "title": "Weaviate: Vector-First Database with Hybrid Search",
        "source": "vector-databases/weaviate-overview.md",
        "text": (
            "Weaviate is an open-source vector database that natively combines vector and scalar storage, "
            "enabling hybrid search without external tooling. Its schema defines classes (analogous to tables) "
            "with properties and vectorizer configurations. The default vectorizer module calls Sentence "
            "Transformers automatically during data ingestion, removing the need for a separate embedding "
            "pipeline. Weaviate's hybrid search API accepts a query string and performs both BM25 lexical "
            "search and vector search internally, fusing results with a user-configurable alpha parameter "
            "(0 = pure BM25, 1 = pure vector). Under the hood, Weaviate uses HNSW for vector search with "
            "configurable ef and maxConnections parameters. The Modules system supports LLM integrations for "
            "generative search: retrieved documents are passed to an LLM (OpenAI, Cohere, or local) to "
            "synthesize a natural language answer. Replication factor and sharding are configurable at the "
            "class level. Weaviate's GraphQL API provides filtering, aggregation, and neighborhood exploration "
            "in a single query language. Tenancy isolation (multi-tenancy) creates separate namespaces per "
            "tenant within a single cluster, enabling cost-effective SaaS deployments where each customer's "
            "data is logically and physically isolated."
        ),
    },
    {
        "id": "doc_027",
        "title": "HNSW Graph Index: Construction and Search",
        "source": "vector-databases/hnsw-index.md",
        "text": (
            "Hierarchical Navigable Small World (HNSW) graphs are the dominant approximate nearest neighbor "
            "index structure for production vector search. HNSW builds a multi-layer graph where upper layers "
            "are sparse (few long-range connections) and lower layers are dense (many short-range connections). "
            "During search, traversal starts at the top layer's entry point and greedily follows edges to "
            "neighbors closer to the query, descending through layers. Layer assignment follows an exponential "
            "decay probability, naturally creating the small-world property. Construction parameters M "
            "(number of connections per node, typically 16-64) and ef_construction (candidate list size "
            "during build, typically 100-400) control the speed-recall trade-off at index build time. "
            "Query-time parameter ef (dynamic candidate list) controls the search-recall trade-off at "
            "query time — larger ef improves recall at the cost of search time. HNSW achieves >99% recall "
            "at 1ms latency for 1M 768-dimensional vectors on a modern CPU with M=48, ef=200. Memory "
            "footprint is approximately 1.1 × (d × 4 + M × 8) bytes per vector. HNSW supports incremental "
            "inserts without index rebuild but does not support deletes natively — deletions require "
            "marking nodes as deleted and periodic graph defragmentation."
        ),
    },
]
