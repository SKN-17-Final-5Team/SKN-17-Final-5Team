"""
Retrieval Performance Evaluation Script

Evaluates RAG retrieval performance using Recall@K and MRR metrics.
Tests different chunking configurations to find optimal settings.
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from qdrant_rag import QdrantCertificationRAG


class RetrievalEvaluator:
    """Evaluates retrieval performance with Recall@K and MRR metrics."""

    def __init__(self, qa_dataset_path: str):
        """Initialize evaluator with QA dataset.

        Args:
            qa_dataset_path: Path to qa_dataset.json file
        """
        self.qa_dataset_path = qa_dataset_path
        self.qa_dataset = self._load_qa_dataset()

    def _load_qa_dataset(self) -> List[Dict]:
        """Load and validate QA dataset."""
        with open(self.qa_dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"✓ Loaded {len(dataset)} QA pairs from {self.qa_dataset_path}")
        return dataset

    def calculate_recall_at_k(
        self,
        results_log: List[Dict],
        k: int
    ) -> Dict[str, Any]:
        """Calculate Recall@K metric.

        Args:
            results_log: List of query results with expected and retrieved certs
            k: Top-K results to consider

        Returns:
            Dictionary with recall score and per-question breakdown
        """
        hits = 0
        details = []

        for result in results_log:
            top_k_docs = result['retrieved'][:k]
            expected = result['expected']

            # Check if any expected cert is in top K
            hit = any(exp in top_k_docs for exp in expected)
            if hit:
                hits += 1

            details.append({
                'question_id': result['question_id'],
                'hit': hit,
                'expected': expected,
                'retrieved_top_k': top_k_docs
            })

        recall = hits / len(results_log) if results_log else 0.0

        return {
            'recall': recall,
            'hits': hits,
            'total': len(results_log),
            'details': details
        }

    def calculate_mrr(self, results_log: List[Dict]) -> Dict[str, Any]:
        """Calculate Mean Reciprocal Rank (MRR) metric.

        Args:
            results_log: List of query results with expected and retrieved certs

        Returns:
            Dictionary with MRR score and per-question breakdown
        """
        reciprocal_ranks = []
        details = []

        for result in results_log:
            retrieved = result['retrieved']
            expected = result['expected']

            # Find first position of any expected cert
            rank = None
            for i, doc in enumerate(retrieved, start=1):
                if doc in expected:
                    rank = i
                    break

            rr = 1.0 / rank if rank else 0.0
            reciprocal_ranks.append(rr)

            details.append({
                'question_id': result['question_id'],
                'rank': rank,
                'reciprocal_rank': rr,
                'expected': expected,
                'retrieved': retrieved[:5]  # Top 5 for display
            })

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

        return {
            'mrr': mrr,
            'details': details
        }

    def evaluate_configuration(
        self,
        config: Dict[str, Any],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Evaluate a specific RAG configuration.

        Args:
            config: Configuration dictionary with chunk_size, embedding_provider, etc.
            top_k: Number of results to retrieve for each query

        Returns:
            Evaluation results with all metrics
        """
        print(f"\n{'='*80}")
        print(f"Evaluating Configuration: {config['name']}")
        print(f"{'='*80}")
        print(f"  Chunk size: {config['chunk_size']}")
        print(f"  Embedding provider: {config['embedding_provider']}")
        print(f"  Collection: {config['collection_name']}")

        # Initialize RAG system
        print(f"\n→ Initializing RAG system...")
        rag = QdrantCertificationRAG(
            collection_name=config['collection_name'],
            chunk_size=config['chunk_size'],
            chunk_overlap=config.get('chunk_overlap', 100),
            embedding_provider=config['embedding_provider'],
            embedding_model=config.get('embedding_model'),
            use_cloud=True
        )

        # Create collection and index documents
        print(f"→ Creating collection (if needed)...")
        rag.create_collection(recreate=config.get('recreate', False))

        # Check if collection needs indexing
        info = rag.get_collection_info()
        text_field = config.get('text_field', 'auto')  # Get text_field from config

        if info.get('points_count', 0) == 0:
            print(f"→ Indexing documents (text_field={text_field})...")
            rag.load_and_index_documents(
                "output/certifications.jsonl",
                text_field=text_field
            )
            print(f"✓ Indexed {info.get('points_count', 0)} documents")
        else:
            print(f"✓ Collection already has {info.get('points_count', 0)} documents")

        # Build BM25 index if hybrid search is enabled
        use_hybrid = config.get('use_hybrid', False)
        if use_hybrid:
            print(f"→ Building BM25 index for hybrid search...")
            # Load documents for BM25
            import json
            documents = []
            with open("output/certifications.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    documents.append(json.loads(line))
            rag.build_bm25_index(documents, text_field=text_field)

        # Run queries and collect results
        search_mode = "hybrid" if use_hybrid else "semantic"
        print(f"\n→ Running {len(self.qa_dataset)} queries ({search_mode} search)...")
        results_log = []

        for i, qa in enumerate(self.qa_dataset, 1):
            question = qa['question']
            expected_certs = qa['expected_certs']

            # Search (hybrid or semantic)
            if use_hybrid:
                search_results = rag.search_hybrid(
                    question,
                    top_k=top_k,
                    semantic_weight=config.get('semantic_weight', 0.7),
                    bm25_weight=config.get('bm25_weight', 0.3)
                )
            else:
                search_results = rag.search(question, top_k=top_k)

            retrieved_certs = [r['doc']['cert_name'] for r in search_results]

            results_log.append({
                'question_id': qa['id'],
                'question': question,
                'expected': expected_certs,
                'retrieved': retrieved_certs,
                'scores': [r['score'] for r in search_results]
            })

            # Progress indicator
            if i % 5 == 0 or i == len(self.qa_dataset):
                print(f"  Progress: {i}/{len(self.qa_dataset)}")

        # Calculate metrics
        print(f"\n→ Calculating metrics...")
        recall_1 = self.calculate_recall_at_k(results_log, k=1)
        recall_3 = self.calculate_recall_at_k(results_log, k=3)
        recall_5 = self.calculate_recall_at_k(results_log, k=5)
        mrr = self.calculate_mrr(results_log)

        # Print summary
        print(f"\n{'='*80}")
        print(f"RESULTS: {config['name']}")
        print(f"{'='*80}")
        print(f"  Recall@1:  {recall_1['recall']:.2%} ({recall_1['hits']}/{recall_1['total']})")
        print(f"  Recall@3:  {recall_3['recall']:.2%} ({recall_3['hits']}/{recall_3['total']})")
        print(f"  Recall@5:  {recall_5['recall']:.2%} ({recall_5['hits']}/{recall_5['total']})")
        print(f"  MRR:       {mrr['mrr']:.4f}")
        print(f"{'='*80}")

        return {
            'config': config,
            'recall_1': recall_1,
            'recall_3': recall_3,
            'recall_5': recall_5,
            'mrr': mrr,
            'results_log': results_log
        }

    def compare_configurations(
        self,
        configs: List[Dict[str, Any]],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Evaluate and compare multiple configurations.

        Args:
            configs: List of configuration dictionaries
            top_k: Number of results to retrieve for each query

        Returns:
            Comparison results with all evaluations
        """
        print(f"\n{'#'*80}")
        print(f"# RETRIEVAL PERFORMANCE EVALUATION")
        print(f"# Dataset: {self.qa_dataset_path}")
        print(f"# QA Pairs: {len(self.qa_dataset)}")
        print(f"# Configurations: {len(configs)}")
        print(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")

        all_results = []

        for config in configs:
            result = self.evaluate_configuration(config, top_k=top_k)
            all_results.append(result)

        # Print comparison table
        self._print_comparison_table(all_results)

        # Find best configuration
        best_config = max(
            all_results,
            key=lambda x: (x['recall_3']['recall'], x['mrr']['mrr'])
        )

        print(f"\n{'='*80}")
        print(f"🏆 BEST CONFIGURATION: {best_config['config']['name']}")
        print(f"{'='*80}")
        print(f"  Recall@3: {best_config['recall_3']['recall']:.2%}")
        print(f"  MRR: {best_config['mrr']['mrr']:.4f}")
        print(f"  Chunk size: {best_config['config']['chunk_size']}")
        print(f"  Embedding: {best_config['config']['embedding_provider']}")
        print(f"{'='*80}")

        return {
            'timestamp': datetime.now().isoformat(),
            'qa_dataset': self.qa_dataset_path,
            'num_questions': len(self.qa_dataset),
            'all_results': all_results,
            'best_config': best_config['config']['name']
        }

    def _print_comparison_table(self, all_results: List[Dict]):
        """Print comparison table of all configurations."""
        print(f"\n{'='*80}")
        print(f"COMPARISON TABLE")
        print(f"{'='*80}")
        print(f"{'Config':<20} {'Recall@1':>12} {'Recall@3':>12} {'Recall@5':>12} {'MRR':>10}")
        print(f"{'-'*80}")

        for result in all_results:
            config_name = result['config']['name']
            r1 = result['recall_1']['recall']
            r3 = result['recall_3']['recall']
            r5 = result['recall_5']['recall']
            mrr = result['mrr']['mrr']

            print(f"{config_name:<20} {r1:>11.2%} {r3:>11.2%} {r5:>11.2%} {mrr:>10.4f}")

        print(f"{'='*80}")

    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save JSON file
        """
        # Remove large details to keep file manageable
        results_copy = json.loads(json.dumps(results))  # Deep copy

        for result in results_copy.get('all_results', []):
            # Keep only summary metrics, remove detailed logs
            if 'results_log' in result:
                del result['results_log']
            for metric in ['recall_1', 'recall_3', 'recall_5', 'mrr']:
                if metric in result and 'details' in result[metric]:
                    del result[metric]['details']

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_path}")


def main():
    """Main evaluation script."""

    # Configuration to test
    configs = [
        {
            "name": "no_chunk_openai",
            "collection_name": "certifications_no_chunk_openai",
            "chunk_size": None,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
            "recreate": False
        },
        {
            "name": "chunk_500_openai",
            "collection_name": "certifications_chunk_500_openai",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
            "recreate": False
        },
        {
            "name": "chunk_1000_openai",
            "collection_name": "certifications_chunk_1000_openai",
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
            "recreate": False
        },
        {
            "name": "chunk_2000_openai",
            "collection_name": "certifications_chunk_2000_openai",
            "chunk_size": 2000,
            "chunk_overlap": 200,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
            "recreate": False
        },
        {
            "name": "no_chunk_hf",
            "collection_name": "certifications_no_chunk_hf",  # Fixed: unique name
            "chunk_size": None,
            "embedding_provider": "huggingface",
            "embedding_model": None,  # Uses jhgan/ko-sroberta-multitask
            "recreate": True  # Force recreate to ensure clean data
        },
        {
            "name": "chunk_1000_hf",
            "collection_name": "certifications_chunk_1000_hf_v2",  # Changed to avoid conflict
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "embedding_provider": "huggingface",
            "embedding_model": None,  # Uses jhgan/ko-sroberta-multitask
            "recreate": True  # Force recreate to ensure clean data
        },
        # Test with full text instead of summary
        {
            "name": "no_chunk_openai_fulltext",
            "collection_name": "certifications_no_chunk_openai_fulltext",
            "chunk_size": None,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
            "text_field": "full",  # Use full cert_subject
            "recreate": False
        },
        {
            "name": "chunk_1000_openai_fulltext",
            "collection_name": "certifications_chunk_1000_openai_fulltext",
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
            "text_field": "full",  # Use full cert_subject
            "recreate": False
        },
        # Test hybrid search (semantic + BM25)
        {
            "name": "no_chunk_openai_fulltext_hybrid",
            "collection_name": "certifications_no_chunk_openai_fulltext",  # Reuse existing
            "chunk_size": None,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
            "text_field": "full",
            "use_hybrid": True,  # Enable hybrid search
            "semantic_weight": 0.7,
            "bm25_weight": 0.3,
            "recreate": False
        },
        {
            "name": "chunk_1000_openai_fulltext_hybrid",
            "collection_name": "certifications_chunk_1000_openai_fulltext",  # Reuse existing
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
            "text_field": "full",
            "use_hybrid": True,  # Enable hybrid search
            "semantic_weight": 0.7,
            "bm25_weight": 0.3,
            "recreate": False
        }
    ]

    # Initialize evaluator
    evaluator = RetrievalEvaluator("qa_dataset.json")

    # Run evaluation
    results = evaluator.compare_configurations(configs, top_k=10)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.json"
    evaluator.save_results(results, output_file)

    print(f"\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
