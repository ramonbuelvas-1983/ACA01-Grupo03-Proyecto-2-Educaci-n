"""Pruebas de segmentación, integridad de cohortes y relevancia de pasajes."""
import unittest
import re
import json
from pathlib import Path
from src.pipeline import compose_chunks, StudentEvidencePipeline, fold, ranked_candidates, detect_source_scope, ensure_source_payload_index

class WhitespaceTokenizer:
    def __call__(self,text,**kwargs):
        matches=list(re.finditer(r"\S+",text))
        return {"input_ids":list(range(len(matches))),"offset_mapping":[m.span() for m in matches]}

class PipelineTests(unittest.TestCase):
    def test_paragraphs_remain_together(self):
        paragraphs=[" ".join(["aprendizaje"]*180)," ".join(["tutoria"]*40)," ".join(["acompanamiento"]*100)]
        chunks=list(compose_chunks(paragraphs,WhitespaceTokenizer()))
        self.assertEqual([c["token_count"] for c in chunks],[220,140])
        self.assertTrue(chunks[1]["text"].startswith(paragraphs[1]))
        text="\n\n".join(paragraphs)
        for chunk in chunks:
            self.assertEqual(chunk["text"],text[chunk["char_start"]:chunk["char_end"]])
    def test_long_paragraph_covers_all_text(self):
        paragraph=" ".join(f"palabra{i}" for i in range(700))
        chunks=list(compose_chunks([paragraph],WhitespaceTokenizer()))
        self.assertTrue(all(c["token_count"]<=280 for c in chunks))
        self.assertEqual(chunks[0]["char_start"],0)
        self.assertEqual(chunks[-1]["char_end"],len(paragraph))
        self.assertTrue(all(b["char_start"]<a["char_end"] for a,b in zip(chunks,chunks[1:])))
    def test_page_only_is_insufficient(self):
        gold={"document_id":"men","pages":[3],"evidence_regex":"alertas tempranas"}
        hit={"doc_id":"men","page":3,"text":"Descripción del presupuesto institucional."}
        self.assertFalse(StudentEvidencePipeline._relevance(hit,gold))
        hit["text"]="Detección mediante ALERTAS  TEMPRANAS."
        self.assertTrue(StudentEvidencePipeline._relevance(hit,gold))
    def test_cloud_payload_index_creation_and_idempotency(self):
        from types import SimpleNamespace
        from qdrant_client import models
        class Client:
            def __init__(self):
                self.schema={}
                self.creates=[]
            def get_collection(self,name):
                return SimpleNamespace(payload_schema=self.schema.copy())
            def create_payload_index(self,**kwargs):
                self.creates.append(kwargs)
                self.schema[kwargs["field_name"]]=SimpleNamespace(data_type=kwargs["field_schema"])
        client=Client()
        first=ensure_source_payload_index(client,"trajectory_test")
        self.assertTrue(first["verified"])
        self.assertTrue(first["created"])
        self.assertEqual(client.creates,[{"collection_name":"trajectory_test","field_name":"doc_id",
                                          "field_schema":models.PayloadSchemaType.KEYWORD,"wait":True}])
        second=ensure_source_payload_index(client,"trajectory_test")
        self.assertFalse(second["created"])
        self.assertEqual(len(client.creates),1)
        client.schema["doc_id"]=SimpleNamespace(data_type=models.PayloadSchemaType.INTEGER)
        with self.assertRaises(ValueError):
            ensure_source_payload_index(client,"trajectory_test")
        self.assertEqual(len(client.creates),1)
    def test_explicit_source_aliases_and_word_boundaries(self):
        docs=[{"id":"ministry","author":"Ministerio de Educación Nacional"},
              {"id":"bosque","author":"Universidad El Bosque"},
              {"id":"national","institution":"Universidad Nacional de Colombia"}]
        self.assertEqual(detect_source_scope("¿Qué propone el MEN?",docs)["document_id"],"ministry")
        self.assertEqual(detect_source_scope("MINISTERIO DE EDUCACION NACIONAL",docs)["document_id"],"ministry")
        self.assertEqual(detect_source_scope("Política de El Bosque",docs)["document_id"],"bosque")
        self.assertEqual(detect_source_scope("Apoyos UNAL",docs)["document_id"],"national")
        self.assertIsNone(detect_source_scope("Un elemento de permanencia",docs)["document_id"])
        multiple=detect_source_scope("Compare MEN y Universidad El Bosque",docs)
        self.assertEqual(multiple["mode"],"multiple_sources")
        self.assertIsNone(multiple["document_id"])
    def test_reranking_preserves_cosine(self):
        dense=[{"dense_rank":1,"score":.91,"text":"A"},{"dense_rank":2,"score":.82,"text":"B"}]
        hits,candidates=ranked_candidates(dense,[-2.,3.],2)
        self.assertEqual([h["text"] for h in hits],["B","A"])
        self.assertEqual(hits[0]["score"],.82)
        self.assertEqual(hits[0]["dense_rank"],2)
        self.assertEqual(hits[0]["rerank_score"],3.)
        self.assertEqual(dense[0]["score"],.91)
        self.assertNotIn("rerank_score",dense[0])
    def test_reranker_rejects_invalid_scores(self):
        with self.assertRaises(ValueError):
            ranked_candidates([{"dense_rank":1,"score":.5}],[],1)
        with self.assertRaises(ValueError):
            ranked_candidates([{"dense_rank":1,"score":.5}],[float("nan")],1)
    def test_accent_normalization(self):
        self.assertEqual(fold("Apoyo ACADÉMICO"),"apoyo academico")
    def test_serialized_gold_anchors_match_their_evidence(self):
        """Detecta escapes JSON que convierten \\s en una barra literal."""
        path=Path(__file__).resolve().parents[1]/"data/evaluation.json"
        benchmark=json.loads(path.read_text(encoding="utf-8"))
        for question in benchmark["questions"]:
            for gold in question["gold"]:
                with self.subTest(question=question["id"]):
                    hit={"doc_id":gold["document_id"],"page":gold["pages"][0],
                         "text":gold["evidence_text"]}
                    self.assertTrue(StudentEvidencePipeline._relevance(hit,gold))

if __name__=="__main__": unittest.main()
