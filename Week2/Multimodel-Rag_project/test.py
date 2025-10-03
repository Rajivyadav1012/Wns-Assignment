"""
End-to-End Integration Test
Step 14: Test all modules working together
"""
import sys
from pathlib import Path

print("\n" + "="*70)
print("STEP 14: END-TO-END INTEGRATION TEST")
print("="*70 + "\n")

errors = []
warnings = []
test_results = {}

# Test 1: Configuration
print("1. Testing Configuration...")
try:
    from config.settings import settings
    print(f"   Base directory: {settings.BASE_DIR}")
    print(f"   Vector DB path: {settings.VECTOR_DB_PATH}")
    api_keys = settings.validate_api_keys()
    configured_keys = [k for k, v in api_keys.items() if v]
    print(f"   Configured APIs: {', '.join(configured_keys) if configured_keys else 'None'}")
    test_results['config'] = 'PASS'
    print("   PASS\n")
except Exception as e:
    print(f"   FAIL: {e}\n")
    errors.append(f"Config: {e}")
    test_results['config'] = 'FAIL'

# Test 2: Logging & Observability
print("2. Testing Logging & Observability...")
try:
    from utils.logger import setup_logger
    from utils.observability import observability
    test_logger = setup_logger("integration_test")
    test_logger.info("Test log message")
    observability.log_event("integration_test", {"test": "value"})
    test_results['logging'] = 'PASS'
    print("   PASS\n")
except Exception as e:
    print(f"   FAIL: {e}\n")
    errors.append(f"Logging: {e}")
    test_results['logging'] = 'FAIL'

# Test 3: Database
print("3. Testing Databases...")
try:
    from utils.database import vector_db, memory_db
    stats = vector_db.get_stats()
    print(f"   Vector DB: {stats['total_vectors']} vectors")
    docs = memory_db.get_all_documents()
    print(f"   Memory DB: {len(docs)} documents")
    test_results['database'] = 'PASS'
    print("   PASS\n")
except Exception as e:
    print(f"   FAIL: {e}\n")
    errors.append(f"Database: {e}")
    test_results['database'] = 'FAIL'

# Test 4: LLM Providers
print("4. Testing LLM Providers...")
try:
    from modules.llm_providers import LLMFactory
    available = LLMFactory.get_available_providers()
    print(f"   Available providers: {', '.join(available)}")
    
    if available:
        # Test with first available provider
        provider = LLMFactory.create_provider(available[0].lower())
        response = provider.generate("Say 'test'", max_tokens=10)
        print(f"   Test response: {response[:50]}...")
        test_results['llm'] = 'PASS'
        print("   PASS\n")
    else:
        print("   WARN: No providers configured\n")
        warnings.append("No LLM providers configured")
        test_results['llm'] = 'WARN'
except Exception as e:
    print(f"   FAIL: {e}\n")
    errors.append(f"LLM: {e}")
    test_results['llm'] = 'FAIL'

# Test 5: Document Processing
print("5. Testing Document Processing...")
try:
    from modules.document_processor import document_processor
    
    # Create test document if needed
    test_file = settings.DOCUMENTS_PATH / "integration_test.txt"
    if not test_file.exists():
        test_file.write_text("This is an integration test document for the RAG system.")
    
    result = document_processor.process_document(
        str(test_file),
        "integration_test.txt"
    )
    
    if result['success']:
        print(f"   Document processed: {result['chunks']} chunks")
        test_results['document_processing'] = 'PASS'
        print("   PASS\n")
    else:
        print(f"   FAIL: {result.get('error')}\n")
        errors.append(f"Document processing failed")
        test_results['document_processing'] = 'FAIL'
except Exception as e:
    print(f"   FAIL: {e}\n")
    errors.append(f"Document Processing: {e}")
    test_results['document_processing'] = 'FAIL'

# Test 6: Standard RAG
print("6. Testing Standard RAG...")
try:
    from modules.rag_standard import StandardRAG
    
    if vector_db.get_stats()['total_vectors'] > 0:
        rag = StandardRAG(
            provider_name="groq" if "Groq" in available else available[0].lower(),
            top_k=2,
            similarity_threshold=0.3
        )
        
        response = rag.query("What is this about?", max_tokens=50)
        print(f"   Retrieved: {response['retrieved_count']} docs")
        print(f"   Time: {response['total_time']:.2f}s")
        test_results['standard_rag'] = 'PASS'
        print("   PASS\n")
    else:
        print("   SKIP: No documents in database\n")
        test_results['standard_rag'] = 'SKIP'
except Exception as e:
    print(f"   FAIL: {e}\n")
    errors.append(f"Standard RAG: {e}")
    test_results['standard_rag'] = 'FAIL'

# Test 7: Hybrid RAG
print("7. Testing Hybrid RAG...")
try:
    from modules.rag_hybrid import HybridRAG
    
    if vector_db.get_stats()['total_vectors'] > 0:
        hybrid = HybridRAG(
            provider_name="groq" if "Groq" in available else available[0].lower(),
            top_k=2,
            similarity_threshold=0.3
        )
        
        # Build BM25 index
        from modules.document_processor import EmbeddingGenerator
        emb_gen = EmbeddingGenerator()
        qe = emb_gen.generate_single_embedding("test")
        all_docs = vector_db.search(qe, 100)
        hybrid.build_bm25_index(all_docs)
        
        response = hybrid.query("What is this about?", max_tokens=50)
        print(f"   Retrieved: {response['retrieved_count']} docs")
        test_results['hybrid_rag'] = 'PASS'
        print("   PASS\n")
    else:
        print("   SKIP: No documents in database\n")
        test_results['hybrid_rag'] = 'SKIP'
except Exception as e:
    print(f"   FAIL: {e}\n")
    errors.append(f"Hybrid RAG: {e}")
    test_results['hybrid_rag'] = 'FAIL'

# Test 8: Knowledge Graph RAG
print("8. Testing Knowledge Graph RAG...")
try:
    from modules.rag_knowledge_graph import KnowledgeGraphRAG
    
    if vector_db.get_stats()['total_vectors'] > 0:
        kg_rag = KnowledgeGraphRAG(
            provider_name="groq" if "Groq" in available else available[0].lower(),
            top_k=2,
            similarity_threshold=0.3
        )
        
        # Build knowledge graph
        from modules.document_processor import EmbeddingGenerator
        emb_gen = EmbeddingGenerator()
        qe = emb_gen.generate_single_embedding("test")
        all_docs = vector_db.search(qe, 100)
        kg_rag.build_graph_from_documents(all_docs)
        
        print(f"   KG entities: {kg_rag.graph.number_of_nodes()}")
        test_results['kg_rag'] = 'PASS'
        print("   PASS\n")
    else:
        print("   SKIP: No documents in database\n")
        test_results['kg_rag'] = 'SKIP'
except Exception as e:
    print(f"   FAIL: {e}\n")
    errors.append(f"KG RAG: {e}")
    test_results['kg_rag'] = 'FAIL'

# Test 9: RAG Selector
print("9. Testing RAG Selector...")
try:
    from modules.rag_selector import RAGSelector
    
    if vector_db.get_stats()['total_vectors'] > 0:
        selector = RAGSelector(
            rag_type="Standard RAG",
            provider_name="groq" if "Groq" in available else available[0].lower(),
            top_k=2,
            similarity_threshold=0.3
        )
        
        response = selector.query("Test query", max_tokens=50)
        print(f"   RAG type: {response.get('rag_type')}")
        test_results['rag_selector'] = 'PASS'
        print("   PASS\n")
    else:
        print("   SKIP: No documents\n")
        test_results['rag_selector'] = 'SKIP'
except Exception as e:
    print(f"   FAIL: {e}\n")
    errors.append(f"RAG Selector: {e}")
    test_results['rag_selector'] = 'FAIL'

# Test 10: Memory Manager
print("10. Testing Memory Manager...")
try:
    from modules.memory_manager import ConversationMemory
    
    memory = ConversationMemory(max_messages=5)
    memory.add_message("user", "Test message")
    messages = memory.get_recent_messages()
    print(f"   Messages stored: {len(messages)}")
    test_results['memory'] = 'PASS'
    print("   PASS\n")
except Exception as e:
    print(f"   FAIL: {e}\n")
    errors.append(f"Memory: {e}")
    test_results['memory'] = 'FAIL'

# Test 11: Guardrails
print("11. Testing Guardrails...")
try:
    from modules.guardrails import ContentGuardrails
    
    guardrails = ContentGuardrails(enable_guardrails=True)
    
    if guardrails.enable_guardrails:
        result = guardrails.check_toxicity("This is a friendly message")
        print(f"   Toxicity check: {'Enabled' if result['checked'] else 'Disabled'}")
        test_results['guardrails'] = 'PASS'
        print("   PASS\n")
    else:
        print("   WARN: Guardrails not enabled\n")
        warnings.append("Guardrails not enabled")
        test_results['guardrails'] = 'WARN'
except Exception as e:
    print(f"   FAIL: {e}\n")
    errors.append(f"Guardrails: {e}")
    test_results['guardrails'] = 'FAIL'

# Test 12: Web Search
print("12. Testing Web Search...")
try:
    from modules.web_search import WebSearchEngine
    
    search = WebSearchEngine()
    if search.enabled:
        results = search.search("test query", num_results=1)
        print(f"   Web search: Enabled ({len(results)} results)")
        test_results['web_search'] = 'PASS'
        print("   PASS\n")
    else:
        print("   WARN: Web search not enabled (no API key)\n")
        warnings.append("Web search not enabled")
        test_results['web_search'] = 'WARN'
except Exception as e:
    print(f"   FAIL: {e}\n")
    errors.append(f"Web Search: {e}")
    test_results['web_search'] = 'FAIL'

# Summary
print("="*70)
print("INTEGRATION TEST SUMMARY")
print("="*70 + "\n")

# Count results
passed = sum(1 for v in test_results.values() if v == 'PASS')
failed = sum(1 for v in test_results.values() if v == 'FAIL')
warned = sum(1 for v in test_results.values() if v == 'WARN')
skipped = sum(1 for v in test_results.values() if v == 'SKIP')
total = len(test_results)

print(f"Total Tests: {total}")
print(f"  PASSED: {passed}")
print(f"  FAILED: {failed}")
print(f"  WARNINGS: {warned}")
print(f"  SKIPPED: {skipped}\n")

# Detailed results
print("Detailed Results:")
for test_name, result in test_results.items():
    symbol = {
        'PASS': '✓',
        'FAIL': '✗',
        'WARN': '⚠',
        'SKIP': '○'
    }.get(result, '?')
    print(f"  {symbol} {test_name}: {result}")

if errors:
    print(f"\nErrors ({len(errors)}):")
    for error in errors:
        print(f"  - {error}")

if warnings:
    print(f"\nWarnings ({len(warnings)}):")
    for warning in warnings:
        print(f"  - {warning}")

print("\n" + "="*70)

if failed == 0:
    print("ALL CRITICAL TESTS PASSED!")
    print("="*70)
    print("\nYour RAG system is ready!")
    print("\nNext steps:")
    print("  1. Run: streamlit run app.py")
    print("  2. Upload documents")
    print("  3. Start chatting with your documents")
else:
    print("SOME TESTS FAILED - Please fix errors above")
    print("="*70)

print()
sys.exit(0 if failed == 0 else 1)